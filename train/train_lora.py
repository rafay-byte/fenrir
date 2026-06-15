import json
from pathlib import Path
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)

# === CUDA Check ===
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Using:", torch.cuda.get_device_name(0))

# === MODEL PATH ===
MODEL_PATH = Path(r"C:\PY PROJECTS\fenrir\models\TinyLlama-1.1B-Chat-v1.0").resolve()

# === Quantization Config ===
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# === Load Tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_PATH),
    local_files_only=True,
    trust_remote_code=True
)

# === Load Model on GPU ===
model = AutoModelForCausalLM.from_pretrained(
    str(MODEL_PATH),
    quantization_config=bnb_config,
    device_map={"": 0},
    local_files_only=True,
    trust_remote_code=True
)

# === Prepare for LoRA ===
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# === Load and Format Dataset ===
def format_example(example):
    return {
        "text": f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"
    }

with open(r"C:\PY PROJECTS\fenrir\data\qa_dataset.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

formatted_data = [format_example(x) for x in raw_data]
dataset = Dataset.from_list(formatted_data)

# === Manual Split (no eval strategy) ===
dataset = dataset.train_test_split(test_size=0.1, seed=42)

# === Tokenize ===
def tokenize_function(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# === Collator ===
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# === Training Args (Safe for old versions) ===
training_args = TrainingArguments(
    output_dir="./train/lora-out",
    per_device_train_batch_size=10,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    num_train_epochs=10,
    warmup_steps=5,
    logging_steps=5,
    save_steps=100,
    save_total_limit=2,
    fp16=True,
    report_to="none"
)

# === Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

# === Train ===
trainer.train()

# === Save ===
model.save_pretrained("train/lora_adapter")
tokenizer.save_pretrained("train/lora_adapter")
