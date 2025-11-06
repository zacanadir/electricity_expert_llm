# ⚡ FLAN-T5 LoRA Ontario Demo

[![Hugging Face Model](https://img.shields.io/badge/HF-Model-blue)](https://huggingface.co/zacanadir/flan-t5-lora-ontario-electricity)
[![Hugging Face Space](https://img.shields.io/badge/HF-Space-brightgreen)](https://huggingface.co/spaces/zacanadir/Ontario-Elec-Expert)

This repository demonstrates **LoRA fine-tuning of FLAN-T5** on a factual dataset about Ontario’s electricity system, and provides an **interactive Gradio app** to compare the base FLAN-T5 model vs the tuned LoRA model.

---

## 📁 Project Structure

flan-t5-lora-ontario-demo/
- README.md
- requirements.txt # Dependencies
- lora_tuning.py # Script for LoRA fine-tuning
- evaluate_lora.py # Script to evaluate LoRA model
- app.py # Gradio app for interactive demo
- ontario_train.json # Training dataset
- ontario_val.json # Validation dataset

---

## 🚀 Requirements

pip install -r requirements.txt
requirements.txt

example:
transformers
datasets
peft
torch
fuzzywuzzy
gradio

🛠️ LoRA Fine-Tuning
Run the LoRA fine-tuning script on your dataset:

python lora_tuning.py
This will:

1. Load google/flan-t5-large

2. Apply LoRA to the attention layers

3. Train on ontario_train.json

4. Save the adapter to ./flan_t5_lora_ontario_adapter

✅ Evaluation
You can evaluate the LoRA adapter vs base model:

python evaluate_lora.py
This script:

1. Loads both base and tuned models

2. Generates predictions on ontario_val.json

3. Computes exact match / similarity metrics

4. Saves detailed results to lora_eval_results.json

🎨 Interactive Demo
A Gradio app allows you to:

- Type or select a question about Ontario electricity

- See side-by-side predictions from base vs LoRA-tuned model

- Highlight differences in color

- Optional similarity scores against reference answer

You can try it here:

💻 Hugging Face Space: https://huggingface.co/spaces/zacanadir/Ontario-Elec-Expert


The LoRA adapter is hosted on HF Hub:
📦 Hugging Face Model: https://huggingface.co/zacanadir/flan-t5-lora-ontario-electricity

Load in code:

from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM

model = PeftModel.from_pretrained(
    AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large"),
    "zacanadir/flan-t5-lora-ontario-electricity"
)

🔧 How to Run Locally
Clone the repo:

git clone https://github.com/zacanadir/electricity_expert_llm.git
cd electricity_expert_llm

Install dependencies:

pip install -r requirements.txt

Launch the app:

python app.py
The app will open in your browser and allow interactive comparisons.

📚 References
FLAN-T5

PEFT: Parameter-Efficient Fine-Tuning

Gradio

🏆 Features
LoRA fine-tuned FLAN-T5 on Ontario electricity data

Base vs Tuned side-by-side comparison

Highlight differences in red/green cards

Optional reference similarity scoring

Clean, modern, intuitive interface for first-time users