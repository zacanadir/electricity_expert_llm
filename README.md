# ⚡ FLAN-T5 LoRA Ontario Demo

[![Hugging Face Model](https://img.shields.io/badge/HF-Model-blue)](https://huggingface.co/zacanadir/flan-t5-lora-ontario-electricity)
[![Hugging Face Space](https://img.shields.io/badge/HF-Space-brightgreen)](https://huggingface.co/spaces/zacanadir/Ontario-Elec-Expert)

This repository demonstrates **LoRA fine-tuning of FLAN-T5** on a factual dataset about Ontario’s electricity system, and provides an **interactive Gradio app** to compare the base FLAN-T5 model vs the tuned LoRA model.

---

## 📁 Project Structure

flan-t5-lora-ontario-demo/
│
├─ README.md
├─ requirements.txt # Dependencies
├─ lora_tuning.py # Script for LoRA fine-tuning
├─ evaluate_lora.py # Script to evaluate LoRA model
├─ app.py # Gradio app for interactive demo
├─ ontario_train.json # Training dataset
├─ ontario_val.json # Validation dataset

---

## 🚀 Requirements

```bash
pip install -r requirements.txt
requirements.txt example:

transformers
datasets
peft
torch
fuzzywuzzy
gradio
🛠️ LoRA Fine-Tuning
Run the LoRA fine-tuning script on your dataset:

bash
python lora_tuning.py
This will:

Load google/flan-t5-large

Apply LoRA to the attention layers

Train on ontario_train.json

Save the adapter to ./flan_t5_lora_ontario_adapter

✅ Evaluation
You can evaluate the LoRA adapter vs base model:

bash

python evaluate_lora.py
This script:

Loads both base and tuned models

Generates predictions on ontario_val.json

Computes exact match / similarity metrics

Saves detailed results to lora_eval_results.json

🎨 Interactive Demo
A Gradio app allows you to:

Type or select a question about Ontario electricity

See side-by-side predictions from base vs LoRA-tuned model

Highlight differences in color

Optional similarity scores against reference answer

You can try it here:

💻 Hugging Face Space

📦 Hugging Face Model
The LoRA adapter is hosted on HF Hub:

🤗 LoRA Adapter Model

Load in code:

python
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM

model = PeftModel.from_pretrained(
    AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large"),
    "your-username/flan-t5-lora-ontario"
)
🔧 How to Run Locally
Clone the repo:

bash

git clone https://github.com/your-username/flan-t5-lora-ontario-demo.git
cd flan-t5-lora-ontario-demo
Install dependencies:

bash

pip install -r requirements.txt
Launch the app:

bash

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