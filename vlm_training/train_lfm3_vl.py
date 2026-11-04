import json
import pathlib

from peft import LoraConfig, get_peft_model
from swift import get_model_processor, get_template, load_dataset, EncodePreprocessor
from swift.trainers import Seq2SeqTrainer, Seq2SeqTrainingArguments


def load_dataset_from_jsonl_files(dataset_paths: list[pathlib.Path]) -> list[dict]:
    datasets = []
    
    for path in dataset_paths:
        if not path.exists():
            raise FileNotFoundError(f"Dataset path {path} does not exist.")
        with open(path, "r") as f:
            lines = f.readlines()
            datasets.extend(list(map(json.loads, lines)))

    return datasets


# Retrieve the model and template, and add a trainable LoRA module
model, tokenizer = get_model_processor(model_id_or_path, ...)
template = get_template(tokenizer, ...)
lora_config = LoraConfig(...)
model = get_peft_model(model, lora_config)

# Download and load the dataset, and encode the text into tokens
train_dataset, val_dataset = load_dataset(dataset_id_or_path, ...)
train_dataset = EncodePreprocessor(template=template)(train_dataset, num_proc=num_proc)
val_dataset = EncodePreprocessor(template=template)(val_dataset, num_proc=num_proc)

# Train the model
training_args = Seq2SeqTrainingArguments(...)
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    template=template,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)
trainer.train()