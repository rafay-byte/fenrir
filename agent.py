# agent.py

import sys
import json
import os
from datetime import datetime
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)
from peft import PeftModel

# === Local model + adapter paths ===
base_model_path = "models/TinyLlama-1.1B-Chat-v1.0"  # Path to quantized TinyLlama model
adapter_path = "train/lora_adapter"  # Path to trained LoRA adapter

# === Load tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(adapter_path)

# === Quantization config for 4-bit loading ===
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# === Load quantized base model ===
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True
)

# === Load LoRA adapter ===
model = PeftModel.from_pretrained(
    model,
    adapter_path,
    local_files_only=True
)

# === Text generation pipeline (no `device=` argument!) ===
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)

# === Read instruction from CLI ===
if len(sys.argv) < 2:
    print("Usage: python agent.py \"<instruction>\"")
    sys.exit(1)

instruction = sys.argv[1]
prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

# === Generate response ===
generated = pipe(prompt)[0]["generated_text"]
response = generated.split("### Response:")[-1].strip()

# === Dry-run simulation for shell-style commands ===
lines = response.split("\n")
dry_run_output = []

if lines[0].startswith(("cd", "git", "ls", "mkdir", "tar", "curl", "rm", "echo", "python", "./", "grep")):
    dry_run_output.append(f"Dry-run: echo {lines[0]}")
    print(f"\n{dry_run_output[-1]}\n")

# === Print final response ===
print("Generated Plan:\n")
print(response)

# === Save logs ===
os.makedirs("logs", exist_ok=True)
with open("logs/trace.jsonl", "a", encoding="utf-8") as logf:
    logf.write(json.dumps({
        "timestamp": datetime.now().isoformat(),
        "instruction": instruction,
        "response": response,
        "dry_run": dry_run_output
    }, ensure_ascii=False) + "\n")
