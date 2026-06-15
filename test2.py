from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

model_name = "microsoft/phi-2"
base_model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = PeftModel.from_pretrained(base_model, "train/lora_adapter")

prompt = "### Instruction:\nHow do I create and switch to a new Git branch?\n\n### Response:\n"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
