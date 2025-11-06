from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model
import torch

# =========================================================
# 1. Device and model setup
# =========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "google/flan-t5-large"

print(f"Loading model on {DEVICE}...")
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32 ,
    device_map="auto" if torch.cuda.is_available() else None
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# =========================================================
# 2. Load dataset
# =========================================================
dataset = load_dataset(
    "json",
    data_files={
        "train": "ontario_train.json",
        "validation": "ontario_val.json"
    }
)

# =========================================================
# 3. Preprocessing
# =========================================================
def preprocess(batch):
    # Build prompts
    prompts = [
        instr + " " + inp
        for instr, inp in zip(batch["instruction"], batch["input"])
    ]

    # Tokenize inputs without pre-padding
    model_inputs = tokenizer(
        prompts,
        max_length=512,
        truncation=True,        
        padding=False           # padding will be efficiently taken care of by collator to save memory
    )

    # Tokenize labels without pre-padding
    labels = tokenizer(
        batch["output"],
        max_length=256,
        truncation=True,
        padding=False           # padding will be efficiently taken care of by collator to save memory
    )

    # Attach labels
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


print("Tokenizing dataset...")
tokenized = dataset.map(preprocess, batched=True, remove_columns=dataset["train"].column_names)

# =========================================================
# 4. LoRA configuration
# =========================================================
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q", "v"],  # attention projection layers in T5
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model = model.to(DEVICE)

# =========================================================
# 5. Training arguments
# =========================================================
args = TrainingArguments(
    output_dir="./flan_t5_lora_ontario",   # where checkpoints and outputs are stored
    overwrite_output_dir=True,             # overwrite existing folder if rerunning experiments
    per_device_train_batch_size=2,         # small batch due to GPU memory
    per_device_eval_batch_size=8,          # can evaluate larger batches (no backprop)
    gradient_accumulation_steps=4,         # effective batch size = 2*4 = 8
    learning_rate=5e-3,                    # standard for LoRA fine-tuning
    num_train_epochs=10,                   # based on dataset size
    save_strategy="epoch",                  # save checkpoints at end of each epoch
    save_total_limit=2,                     # keep only last 2 checkpoints to save space
    logging_steps=50,                       # log metrics every 50 steps
    eval_strategy="epoch",                  # evaluate once per epoch
    load_best_model_at_end=True,            # restore the checkpoint with best metric
    metric_for_best_model="loss",           # monitor validation loss
    greater_is_better=False,                # lower loss is better
    fp16=False,                              # mixed precision for memory and speed
    dataloader_num_workers=2,               # number of CPU workers for data loading
    seed=42,                                # ensures reproducibility
    report_to="none",                        # no automatic reporting to WandB/TensorBoard
)

# =========================================================
# 6. Data collator
# =========================================================
collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# =========================================================
# 7. Trainer setup
# =========================================================
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    data_collator=collator,
)

# =========================================================
# 8. Training
# =========================================================
print("Starting fine-tuning...")
trainer.train()

# =========================================================
# 9. Save LoRA adapter only
# =========================================================
model.save_pretrained("./flan_t5_lora_ontario_adapter", save_adapter=True)
print("✅ LoRA adapter saved to ./flan_t5_lora_ontario_adapter")
