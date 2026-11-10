"""
Fine-tune LiquidAI/LFM3-VL-2B with a plain PyTorch training loop.

- transformers is used ONLY to load weights + processor (no Trainer, no TRL, no PEFT).
- Two modes:
    MODE="lora": hand-written LoRA on nn.Linear layers, frozen bf16 base  (~10-16 GB VRAM)
    MODE="full": full fine-tune, fp32 master weights + bf16 autocast    (~40 GB+ VRAM)
- Loss only on assistant tokens (prompt / image / padding tokens masked with -100).

Data: JSONL, one sample per line:
    {"image": "path/to/img.png", "prompt": "What is shown?", "response": "A lamella ..."}

Requires: pip install "transformers>=5.21.0" torch torchvision pillow
"""
import abc
import json
import math
import os
import random
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForImageTextToText, AutoProcessor

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
QWEN_BOX_NORMALIZATION_FACTOR = 1000

# ----------------------------------------------------------------------------- config
@dataclass
class Cfg:
    model_id: str = "LiquidAI/LFM3-VL-2B"
    train_jsonl: str = "dataset.jsonl"
    val_jsonl: str = "dataset.jsonl"
    out_dir: str = "out/lfm3-vl-2b-ft"

    mode: str = "lora"              # "lora" | "full"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    freeze_vision: bool = True      # don't touch the 164M vision encoder
    train_projector: bool = False    # fully train the 25M projector (cheap, usually helps)
    # substrings used to recognise modules; check the printed module tree and adjust
    vision_keys: tuple = ("vision",)
    projector_keys: tuple = ("projector",)

    max_image_side: int = 2048       # LFM3-VL runs at native res -> cap tokens/memory here
    max_len: int = 4096             # drop samples longer than this (after image expansion)

    inject_think: bool = True     # prepend a layout <think> trace synthesized from GT boxes

    epochs: int = 15
    batch_size: int = 2
    grad_accum: int = 4
    lr: float = 2e-4                # ~2e-4 for LoRA, ~1e-5 for full FT
    weight_decay: float = 0.0
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    grad_ckpt: bool = True
    eval_every: int = 200           # optimizer steps
    num_workers: int = 4
    seed: int = 0


cfg = Cfg()


def _clip01(v: float) -> float:
    return min(1.0, max(0.0, v))


@dataclass
class Box:
    """Axis-aligned box; augmentations assume coords normalized to 0-1."""
    x1: float
    y1: float
    x2: float
    y2: float
    label: str = "product"
    
    def __post_init__(self):
        if self.x1 > self.x2 or self.y1 > self.y2:
            raise ValueError(f"Box coordinates must be valid: {self}")
        
    def jitter(self, max_jitter: float) -> "Box":
        x1, x2 = (_clip01(v + random.uniform(-max_jitter, max_jitter)) for v in (self.x1, self.x2))
        y1, y2 = (_clip01(v + random.uniform(-max_jitter, max_jitter)) for v in (self.y1, self.y2))
        return Box(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2), self.label)
        
    def crop(self, start_x: float, start_y: float, end_x: float, end_y: float) -> "Box | None":
        """Re-express the box in the normalized frame of the crop window
        (window given in normalized coords of the current image). Returns None
        if the box falls outside the window."""
        w, h = end_x - start_x, end_y - start_y
        x1 = _clip01((self.x1 - start_x) / w)
        y1 = _clip01((self.y1 - start_y) / h)
        x2 = _clip01((self.x2 - start_x) / w)
        y2 = _clip01((self.y2 - start_y) / h)
        if x2 <= x1 or y2 <= y1:
            return None
        return Box(x1, y1, x2, y2, self.label)
    
    def scale(self, scale_x: float, scale_y: float) -> "Box":
        return Box(self.x1 * scale_x, self.y1 * scale_y, self.x2 * scale_x, self.y2 * scale_y, self.label)


class Augmentation(abc.ABC):
    @abc.abstractmethod
    def __call__(self, image: Image.Image, boxes: list[Box]) -> tuple[Image.Image, list[Box]]:
        pass


class RandomCrop(Augmentation):
    def __init__(self, p: float = 0.5, max_crop_ratio: float = 0.1):
        self.p = p
        self.max_crop_ratio = max_crop_ratio
    
    def __call__(self, image: Image.Image, boxes: list[Box]) -> tuple[Image.Image, list[Box]]:
        if random.random() > self.p:
            return image, boxes
        
        W, H = image.size
        mx, my = int(W * self.max_crop_ratio), int(H * self.max_crop_ratio)
        start_x = random.randint(0, mx)
        start_y = random.randint(0, my)
        end_x = random.randint(W - mx, W)
        end_y = random.randint(H - my, H)
        
        image = image.crop((start_x, start_y, end_x, end_y))
        
        # window normalized by the ORIGINAL size; boxes get re-normalized to the crop
        window = (start_x / W, start_y / H, end_x / W, end_y / H)
        new_boxes = [b for b in (box.crop(*window) for box in boxes) if b is not None]
        return image, new_boxes


class RandomJitter(Augmentation):
    def __init__(self, p: float = 0.5, max_jitter: float = 0.005):
        self.p = p
        self.max_jitter = max_jitter  # fraction of image size
    
    def __call__(self, image: Image.Image, boxes: list[Box]) -> tuple[Image.Image, list[Box]]:
        if random.random() > self.p:
            return image, boxes
        return image, [box.jitter(self.max_jitter) for box in boxes]


class RandomResize(Augmentation):
    def __init__(self, p: float = 0.5, min_scale: float = 0.8, max_scale: float = 1.2, max_side: int | None = None):
        self.p = p
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.max_side = max_side
    
    def __call__(self, image: Image.Image, boxes: list[Box]) -> tuple[Image.Image, list[Box]]:
        if random.random() > self.p:
            return image, boxes
        
        scale = random.uniform(self.min_scale, self.max_scale)
        if self.max_side:
            scale = min(scale, self.max_side / max(image.size))
        new_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        # uniform resize: normalized box coords are unchanged
        return image.resize(new_size, Image.Resampling.BICUBIC), boxes

    
class Compose(Augmentation):
    def __init__(self, augmentations: list[Augmentation]):
        self.augmentations = augmentations
    
    def _boxes_to_normalized(self, boxes: list[Box]) -> list[Box]:
        return [box.scale(1 / QWEN_BOX_NORMALIZATION_FACTOR, 1 / QWEN_BOX_NORMALIZATION_FACTOR) for box in boxes]
    
    def _boxes_to_original(self, boxes: list[Box]) -> list[Box]:
        return [box.scale(QWEN_BOX_NORMALIZATION_FACTOR, QWEN_BOX_NORMALIZATION_FACTOR) for box in boxes]
    
    def __call__(self, image: Image.Image, boxes: list[Box]) -> tuple[Image.Image, list[Box]]:
        boxes = self._boxes_to_normalized(boxes)
        for augmentation in self.augmentations:
            image, boxes = augmentation(image, boxes)
            
        boxes = self._boxes_to_original(boxes)
        return image, boxes



# ----------------------------------------------------------------------------- data
def _synth_think(answer: str) -> str:
    """Short reasoning trace describing the GT layout, so the model learns to
    analyse the page before emitting coordinates."""
    from vlm_training.utils.bbox_viz import parse_bboxes
    try:
        boxes = parse_bboxes(answer)
    except (ValueError, KeyError):
        return "<think>I will locate every product in the leaflet.</think>"
    # group boxes into rows by vertical centre
    rows = []
    for b in sorted(boxes, key=lambda d: (d["bbox_2d"][1] + d["bbox_2d"][3]) / 2):
        cy = (b["bbox_2d"][1] + b["bbox_2d"][3]) / 2
        if rows and cy - rows[-1][0] < 150:
            rows[-1][1].append(b)
        else:
            rows.append([cy, [b]])
    per_row = ", ".join(f"{len(r[1])}" for r in rows)
    return (
        f"<think>The leaflet shows {len(boxes)} products in {len(rows)} row(s) "
        f"({per_row} per row). Reading left to right, top to bottom, I will output "
        f"one box per product covering the item and its price.</think>"
    )


class JsonlVLDataset(Dataset):
    def __init__(self, path, max_side, inject_think=False, augment=False):
        with open(path) as f:
            self.rows = [json.loads(l) for l in f if l.strip()]
        self.max_side = max_side
        self.inject_think = inject_think
        self.augmentations = Compose([
            RandomCrop(p=0.5, max_crop_ratio=0.1),
            RandomResize(p=0.5, min_scale=0.8, max_scale=1.2, max_side=max_side),
            RandomJitter(p=0.5, max_jitter=0.02),
        ]) if augment else None

    def __len__(self):
        return len(self.rows)
    
    def _parse_boxes(self, answer: str) -> list[Box]:
        from vlm_training.utils.bbox_viz import parse_bboxes
        try:
            boxes = parse_bboxes(answer)
        except (ValueError, KeyError):
            return []
        return [Box(*b["bbox_2d"], label=b["label"]) for b in boxes]
    
    def _encode_boxes(self, boxes: list[Box]) -> str:
        from vlm_training.utils.bbox_viz import encode_boxes
        return encode_boxes(boxes)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(r["images"][0]).convert("RGB")
        img.thumbnail((self.max_side, self.max_side))  # keeps aspect ratio, only downsizes
        answer = r["conversations"][1]["value"]
        
        if self.augmentations:
            boxes = self._parse_boxes(answer)
            img, boxes = self.augmentations(img, boxes)
            # rebuild the target from the augmented boxes (drop any stale GT think trace)
            answer = self._encode_boxes(boxes)
        
        if self.inject_think and "</think>" not in answer:
            answer = _synth_think(answer) + answer
        # Don't put <image> tags in text: the chat template inserts the vision tokens.
        return [
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": r["conversations"][0]["value"]},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": answer},
            ]},
        ]


class Collator:
    """Tokenise a batch of conversations and build assistant-only labels."""

    def __init__(self, processor, max_len):
        self.p = processor
        self.tok = processor.tokenizer
        self.tok.padding_side = "right"
        self.max_len = max_len
        # ChatML: <|im_start|>assistant\n ... <|im_end|>
        self.header = self.tok.encode("<|im_start|>assistant\n", add_special_tokens=False)
        self.im_end = self.tok.convert_tokens_to_ids("<|im_end|>")

    def _labels(self, input_ids, attn):
        labels = torch.full_like(input_ids, -100)
        n = len(self.header)
        for b in range(input_ids.size(0)):
            ids = input_ids[b].tolist()
            found, i = False, 0
            while i <= len(ids) - n:
                if ids[i:i + n] == self.header:
                    s = j = i + n
                    while j < len(ids) and ids[j] != self.im_end:
                        j += 1
                    e = min(j + 1, len(ids))          # include <|im_end|> so it learns to stop
                    labels[b, s:e] = input_ids[b, s:e]
                    found, i = True, e
                else:
                    i += 1
            if not found:
                raise RuntimeError("assistant header not found; check chat template / header tokens")
        labels[attn == 0] = -100
        return labels

    def __call__(self, convs):
        batch = self.p.apply_chat_template(
            convs,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=False,
            processor_kwargs={"text_kwargs": {"padding": True}},
        )
        if batch["input_ids"].size(1) > self.max_len:
            # Simplest policy: truncate text. Never truncate inside image tokens in real use;
            # better to filter long samples offline.
            raise ValueError(f"batch seq len {batch['input_ids'].size(1)} > max_len {self.max_len}")
        batch = dict(batch)
        batch["labels"] = self._labels(batch["input_ids"], batch["attention_mask"])
        return batch


# ----------------------------------------------------------------------------- LoRA (pure PyTorch)
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float):
        super().__init__()
        self.base = base
        self.scale = alpha / r
        dev = base.weight.device
        # fp32 adapter params; autocast runs the matmuls in bf16
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, device=dev, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    @property
    def in_features(self):
        return self.base.in_features
    
    @property
    def out_features(self):
        return self.base.out_features

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(self.drop(x), self.lora_A), self.lora_B) * self.scale

    @torch.no_grad()
    def merged(self) -> nn.Linear:
        delta = (self.lora_B @ self.lora_A) * self.scale
        self.base.weight += delta.to(self.base.weight.dtype)
        return self.base


def _matches(name, keys):
    return any(k in name for k in keys)


def apply_lora(model, c: Cfg):
    out_emb = model.get_output_embeddings()
    targets = []
    for name, m in model.named_modules():
        if not isinstance(m, nn.Linear) or m is out_emb:
            continue
        if c.freeze_vision and _matches(name, c.vision_keys):
            continue
        if c.train_projector and _matches(name, c.projector_keys):
            continue  # trained fully instead
        targets.append(name)
    for name in targets:
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, LoRALinear(getattr(parent, child), c.lora_r, c.lora_alpha, c.lora_dropout))
    return len(targets)


def merge_lora(model):
    names = [n for n, m in model.named_modules() if isinstance(m, LoRALinear)]
    for name in names:
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, getattr(parent, child).merged())


def setup_trainable(model, c: Cfg):
    if c.mode == "lora":
        for p in model.parameters():
            p.requires_grad_(False)
        n = apply_lora(model, c)
        print(f"LoRA on {n} Linear layers")
        if c.train_projector:
            for name, p in model.named_parameters():
                if _matches(name, c.projector_keys):
                    p.data = p.data.float()       # fp32 master copy for the trainable projector
                    p.requires_grad_(True)
    else:
        for name, p in model.named_parameters():
            p.requires_grad_(not (c.freeze_vision and _matches(name, c.vision_keys)))
    tot = sum(p.numel() for p in model.parameters())
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable {tr/1e6:.1f}M / {tot/1e6:.1f}M ({100*tr/tot:.2f}%)")


# ----------------------------------------------------------------------------- loss
def lm_loss(model, batch):
    labels = batch.pop("labels")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(**batch, use_cache=False).logits
    # shift: token t predicts t+1. CE in fp32 for stability (vocab 128k -> watch memory).
    logits = logits[:, :-1].float()
    labels = labels[:, 1:]
    loss_sum = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
                               ignore_index=-100, reduction="sum")
    return loss_sum, (labels != -100).sum()


def to_device(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tot, n = 0.0, 0
    for batch in loader:
        s, k = lm_loss(model, to_device(batch, device))
        tot += s.item(); n += k.item()
    model.train()
    return tot / max(n, 1)


# ----------------------------------------------------------------------------- main
def main(c: Cfg):
    random.seed(c.seed); torch.manual_seed(c.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")

    processor = AutoProcessor.from_pretrained(c.model_id, trust_remote_code=True, freeze_vision=False)
    load_dtype = torch.bfloat16 if c.mode == "lora" else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(
        c.model_id, dtype=load_dtype, trust_remote_code=True
    ).to(device)

    # Inspect top-level structure once to confirm vision/projector names
    print("top-level modules:", [n for n, _ in model.named_children()])
    for n, ch in model.named_children():
        print("  ", n, "->", [m for m, _ in ch.named_children()][:8])

    setup_trainable(model, c)
    if c.grad_ckpt:
        # non-reentrant -> works with frozen embeddings / LoRA without enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()

    collate = Collator(processor, c.max_len)
    train_ds = JsonlVLDataset(c.train_jsonl, c.max_image_side, c.inject_think, augment=True)
    val_ds = JsonlVLDataset(c.val_jsonl, c.max_image_side, c.inject_think)
    train_dl = DataLoader(train_ds, c.batch_size, shuffle=True, collate_fn=collate,
                          num_workers=c.num_workers, pin_memory=True, drop_last=True,
                          persistent_workers=c.num_workers > 0)
    val_dl = DataLoader(val_ds, c.batch_size, shuffle=False, collate_fn=collate,
                        num_workers=c.num_workers, pin_memory=True)

    # Sanity check: show exactly what is supervised in the first sample
    b0 = collate([train_ds[0]])
    tgt = b0["labels"][0][b0["labels"][0] != -100]
    print("SUPERVISED TEXT >>>", processor.tokenizer.decode(tgt), "<<<")

    # Optimizer: no weight decay on biases / norms / LoRA B
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 or "norm" in n.lower() else decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": c.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=c.lr, betas=(0.9, 0.95), eps=1e-8, fused=True,
    )

    total_steps = math.ceil(len(train_dl) / c.grad_accum) * c.epochs
    warmup = max(1, int(c.warmup_ratio * total_steps))

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog))   # cosine to 10%

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    trainable = [p for p in model.parameters() if p.requires_grad]

    step, best = 0, float("inf")
    os.makedirs(c.out_dir, exist_ok=True)
    t0 = time.time()

    for epoch in range(c.epochs):
        accum_loss, accum_tok = 0.0, 0
        micro = []
        for i, batch in enumerate(train_dl):
            micro.append(batch)
            if len(micro) < c.grad_accum and i != len(train_dl) - 1:
                continue

            # token-weighted loss across micro-batches (correct normalisation for grad accum)
            n_tok = sum((b["labels"][:, 1:] != -100).sum().item() for b in micro)
            for b in micro:
                loss_sum, _ = lm_loss(model, to_device(b, device))
                (loss_sum / n_tok).backward()
                accum_loss += loss_sum.item()
            accum_tok += n_tok
            micro = []

            gnorm = torch.nn.utils.clip_grad_norm_(trainable, c.max_grad_norm)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1

            if step % 10 == 0:
                print(f"ep {epoch} step {step}/{total_steps} loss {accum_loss/accum_tok:.4f} "
                      f"gnorm {gnorm:.2f} lr {sched.get_last_lr()[0]:.2e} "
                      f"mem {torch.cuda.max_memory_allocated()/2**30:.1f}G {time.time()-t0:.0f}s")
                accum_loss, accum_tok = 0.0, 0

            if step % c.eval_every == 0 or step == total_steps:
                vl = evaluate(model, val_dl, device)
                print(f"== step {step} val loss {vl:.4f} ppl {math.exp(vl):.2f}")
                if vl < best:
                    best = vl
                    ckpt = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
                    torch.save(ckpt, os.path.join(c.out_dir, "best_trainable.pt"))

    # ---- export a standalone HF checkpoint
    best_sd = torch.load(os.path.join(c.out_dir, "best_trainable.pt"))
    model.load_state_dict(best_sd, strict=False)
    if c.mode == "lora":
        merge_lora(model)
    model.to(torch.bfloat16)
    if c.grad_ckpt:
        model.gradient_checkpointing_disable()
    model.config.use_cache = True
    model.save_pretrained(os.path.join(c.out_dir, "final"))
    processor.save_pretrained(os.path.join(c.out_dir, "final"))
    print("saved", os.path.join(c.out_dir, "final"))


if __name__ == "__main__":
    main(cfg)
