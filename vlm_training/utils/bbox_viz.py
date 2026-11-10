"""Parse VLM detection responses and draw bounding boxes on images."""

import hashlib
import json
import re

from PIL import Image, ImageDraw, ImageFont


_BOX_KEYS = ("bbox_2d", "box_2d", "bbox", "box", "coords", "coordinates")
_LABEL_KEYS = ("label", "product", "name", "item", "category", "class")


def _loads_tolerant(text: str) -> list:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # LLM typo: "bbox_2d": "1, 2, 3, 4]  (string closed by ] instead of ")
    fixed = re.sub(r'"(\s*[\d.,\s-]+?)\s*\]', r'"\1"', text)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    # last resort: salvage whatever well-formed objects we can
    dets = []
    for m in re.finditer(r"\{[^{}]*\}", fixed):
        try:
            dets.append(json.loads(m.group(0)))
        except json.JSONDecodeError:
            continue
    if dets:
        return dets
    raise ValueError(f"unparseable JSON in response: {text[:200]!r}")


def _norm_bbox(box) -> list[float]:
    if isinstance(box, str):
        box = re.split(r"[,\s]+", box.strip())
    elif len(box) == 1 and isinstance(box[0], str):  # ["1, 2, 3, 4"]
        return _norm_bbox(box[0])
    box = [float(v) for v in box]
    if len(box) != 4:
        raise ValueError(f"bbox must have 4 coords, got {box}")
    return box


def _norm_detection(det) -> dict:
    if isinstance(det, (list, tuple)):
        return {"bbox_2d": _norm_bbox(det), "label": "object"}
    box = None
    for k in _BOX_KEYS:
        if k in det:
            box = det[k]
            break
    if box is None:  # fall back to the first 4-number value
        for v in det.values():
            try:
                box = _norm_bbox(v)
                break
            except (ValueError, TypeError):
                continue
    if box is None:
        raise ValueError(f"no bbox coords found in detection: {det}")
    label = next((det[k] for k in _LABEL_KEYS if k in det), "object")
    return {**det, "bbox_2d": _norm_bbox(box), "label": label}


def parse_bboxes(response: str) -> list[dict]:
    """Extract the bbox list from a model response.

    Strips a leading <think>...</think> block, markdown code fences and
    trailing special tokens, then parses the first JSON array. Coords may
    be a list or a comma-separated string under various key names
    ("bbox_2d", "box_2d", "box", ...); every detection is normalised to
    {"bbox_2d": [x1, y1, x2, y2] floats, "label": str} in 0-1000 space.
    """
    text = response.split("</think>")[-1]
    text = text.replace("`", "").strip()
    if text.startswith("json"):
        text = text[4:].lstrip()
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON array in response: {text[:200]!r}")
    return [_norm_detection(det) for det in _loads_tolerant(match.group(0))]


def encode_boxes(boxes) -> str:
    """Inverse of parse_bboxes: objects with x1, y1, x2, y2 (0-1000) and label
    -> fenced JSON in the dataset's answer format."""
    dets = [{"bbox_2d": [round(b.x1), round(b.y1), round(b.x2), round(b.y2)], "label": b.label} for b in boxes]
    return "```json\n" + json.dumps(dets) + "\n```"


def _color_for_label(label: str) -> tuple[int, int, int]:
    h = hashlib.md5(label.encode()).digest()
    return h[0] % 200 + 55, h[1] % 200 + 55, h[2] % 200 + 55


def draw_bboxes(image: Image.Image, bboxes: list[dict], coord_range: float = 1000.0) -> Image.Image:
    """Draw boxes in relative coords (0-coord_range) onto a copy of image."""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    font = ImageFont.load_default()
    for det in bboxes:
        x1, y1, x2, y2 = det["bbox_2d"]
        box = [x1 / coord_range * w, y1 / coord_range * h,
               x2 / coord_range * w, y2 / coord_range * h]
        color = _color_for_label(det.get("label", "object"))
        draw.rectangle(box, outline=color, width=3)
        label = det.get("label", "")
        if label:
            tw = draw.textlength(label, font=font)
            ty = max(0, box[1] - 14)
            draw.rectangle([box[0], ty, box[0] + tw + 6, ty + 14], fill=color)
            draw.text((box[0] + 3, ty + 1), label, fill="white", font=font)
    return img


def compare_bboxes(image, gt: list[dict], pred: list[dict],
                   coord_range: float = 1000.0) -> Image.Image:
    """Draw GT (green) and predicted (red) boxes on one image."""
    if not isinstance(image, Image.Image):
        image = Image.open(image).convert("RGB")
    img = image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    font = ImageFont.load_default()
    for dets, color, tag in ((gt, (0, 200, 0), "GT"), (pred, (230, 30, 30), "P")):
        for det in dets:
            x1, y1, x2, y2 = det["bbox_2d"]
            box = [x1 / coord_range * w, y1 / coord_range * h,
                   x2 / coord_range * w, y2 / coord_range * h]
            draw.rectangle(box, outline=color, width=2)
            draw.text((box[0] + 2, box[1] + 2), tag, fill=color, font=font)
    return img


def show_bboxes(image, response: str, coord_range: float = 1000.0) -> Image.Image:
    """Parse response and return the image with its boxes drawn on it.

    In a notebook, use as the last expression in a cell to display it;
    in a script, call .show() or .save() on the result. coord_range is
    the model's coordinate normalisation (1000 or e.g. 1024).
    """
    if not isinstance(image, Image.Image):
        image = Image.open(image).convert("RGB")
    return draw_bboxes(image, parse_bboxes(response), coord_range)
