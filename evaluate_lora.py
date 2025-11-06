import json
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
import torch
from tqdm import tqdm
from fuzzywuzzy import fuzz  # optional for fuzzy matching

# =========================================================
# 1. Configuration
# =========================================================
BASE_MODEL = "google/flan-t5-large"
ADAPTER_PATH = "./flan_t5_lora_ontario_adapter"
VAL_DATA = "ontario_val.json"
MAX_TOKENS = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================
# 2. Load tokenizer
# =========================================================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# =========================================================
# 3. Load models
# =========================================================
print("Loading base model...")
base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL).to(DEVICE)
base_model.eval()

print("Loading LoRA adapter...")
lora_base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL).to(DEVICE)
tuned_model = PeftModel.from_pretrained(lora_base_model, ADAPTER_PATH).to(DEVICE)
tuned_model.eval()

# =========================================================
# 4. Load validation dataset
# =========================================================
dataset = load_dataset("json", data_files={"eval": VAL_DATA})["eval"]

# =========================================================
# 5. Helper function to generate answer
# =========================================================
def generate_answer(model, question, max_length=MAX_TOKENS):
    inputs = tokenizer(question, return_tensors="pt").to(DEVICE)
    outputs = model.generate(**inputs, max_new_tokens=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# =========================================================
# 6. Evaluate
# =========================================================
results = []
base_correct = 0
tuned_correct = 0
total = len(dataset)

for example in tqdm(dataset, desc="Evaluating"):
    question = example["instruction"] + " " + example["input"]
    gold = example["output"]

    # Base model prediction
    base_pred = generate_answer(base_model, question)

    # Tuned model prediction
    tuned_pred = generate_answer(tuned_model, question)

    # Exact-match scoring
    base_match = int(base_pred.strip().lower() == gold.strip().lower())
    tuned_match = int(tuned_pred.strip().lower() == gold.strip().lower())

    base_correct += base_match
    tuned_correct += tuned_match

    results.append({
        "question": question,
        "gold": gold,
        "base_pred": base_pred,
        "tuned_pred": tuned_pred,
        "base_match": base_match,
        "tuned_match": tuned_match
    })

# =========================================================
# 7. Compute accuracy
# =========================================================
base_acc = base_correct / total * 100
tuned_acc = tuned_correct / total * 100

print(f"Base model exact-match accuracy: {base_acc:.2f}%")
print(f"Tuned model exact-match accuracy: {tuned_acc:.2f}%")

# =========================================================
# 8. Save detailed results
# =========================================================
with open("lora_eval_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("✅ Evaluation completed. Results saved to lora_eval_results.json")
