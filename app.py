import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from fuzzywuzzy import fuzz
import torch, json, os

# =========================
# 1. Config
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_TOKENS = 128
BASE_MODEL = "google/flan-t5-large"
ADAPTER = "zacanadir/flan-t5-lora-ontario-electricity"  
VAL_FILE = "ontario_val.json"

# =========================
# 2. Load models
# =========================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

def load_model():
    base = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL).to(DEVICE)
    base.eval()
    tuned = PeftModel.from_pretrained(
        AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL),
        ADAPTER
    ).to(DEVICE)
    tuned.eval()
    return base, tuned

base_model, tuned_model = load_model()

# =========================
# 3. Load dataset for examples
# =========================
dataset = []
if os.path.exists(VAL_FILE):
    with open(VAL_FILE, "r", encoding="utf-8") as f:
        text = f.read().strip()
        if text.startswith("["):
            dataset = json.loads(text)
        else:
            dataset = [json.loads(line) for line in text.splitlines() if line.strip()]

example_questions = [ex["instruction"] + " " + ex["input"] for ex in dataset[:20]] if dataset else []
gold_dict = {ex["instruction"] + " " + ex["input"]: ex["output"] for ex in dataset} if dataset else {}

# =========================
# 4. Helper functions
# =========================
def generate_answer(model, question):
    inputs = tokenizer(question, return_tensors="pt").to(DEVICE)
    outputs = model.generate(**inputs, max_new_tokens=MAX_TOKENS)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Card templates
placeholder_card = """
<div style="padding:15px;margin-bottom:10px;border-radius:10px;
     box-shadow:0 2px 5px rgba(0,0,0,0.1);background:#f9f9f9;text-align:center;">
     <i>Your predictions will appear here after clicking 'Compare Predictions'</i>
</div>
"""

spinner_card_template = """
<div style="padding:15px;margin-bottom:10px;border-radius:10px;
     box-shadow:0 2px 5px rgba(0,0,0,0.1);background:#f9f9f9;text-align:center;">
     <div class="loader"></div> {message}
</div>
<style>
.loader {{
  border: 4px solid #f3f3f3;
  border-top: 4px solid #336699;
  border-radius: 50%;
  width: 24px;
  height: 24px;
  animation: spin 1s linear infinite;
  display:inline-block;
  margin-right: 8px;
}}
@keyframes spin {{0%{{transform:rotate(0deg);}}100%{{transform:rotate(360deg);}}}}
</style>
"""

def compare_models(question, show_scores):
    if not question.strip():
        return placeholder_card, placeholder_card, placeholder_card, placeholder_card

    # Show spinners immediately
    yield (spinner_card_template.format(message="Generating Base prediction..."),
           spinner_card_template.format(message="Generating Tuned prediction..."),
           spinner_card_template.format(message="Generating Differences..."),
           spinner_card_template.format(message="Computing similarity scores..."))

    # Generate predictions
    base_pred = generate_answer(base_model, question)
    tuned_pred = generate_answer(tuned_model, question)
    gold = gold_dict.get(question, "")

    base_score = fuzz.token_set_ratio(base_pred, gold) if gold else None
    tuned_score = fuzz.token_set_ratio(tuned_pred, gold) if gold else None

    # Differences
    diff = []
    for b, t in zip(base_pred.split(". "), tuned_pred.split(". ")):
        if b != t:
            diff.append(f"<span style='color:#d62728'><b>Base:</b> {b}</span><br>"
                        f"<span style='color:#2ca02c'><b>Tuned:</b> {t}</span>")
        else:
            diff.append(f"<span style='opacity:0.7'>{b}</span>")
    diff_text = "<br><br>".join(diff)

    # Similarity scores
    if show_scores and gold:
        score_html = f"""
        <div style='text-align:center;margin-top:10px'>
            <b>Reference:</b> {gold}<br><br>
            <div style='display:inline-block;padding:6px 12px;background:#f6f6f6;border-radius:8px'>
                <b>Base similarity:</b> {base_score:.1f}% &nbsp;&nbsp; | &nbsp;&nbsp;
                <b>Tuned similarity:</b> <span style='color:#2ca02c'>{tuned_score:.1f}%</span>
            </div>
        </div>
        """
    else:
        score_html = "<span style='opacity:0.6;'>Scores hidden.</span>" if gold else ""

    # Wrap outputs in cards with headers
    card_style = "padding:15px;margin-bottom:10px;border-radius:10px;box-shadow:0 2px 5px rgba(0,0,0,0.1);background:#f9f9f9;"
    base_box_html = f"<div style='{card_style}'><h4 style='color:#d62728;margin-top:0;'>Base Model</h4>{base_pred}</div>"
    tuned_box_html = f"<div style='{card_style};border-left:4px solid #2ca02c'><h4 style='color:#2ca02c;margin-top:0;'>Tuned Model</h4>{tuned_pred}</div>"
    diff_box_html = f"<div style='{card_style}'><h4 style='margin-top:0;'>Differences Highlighted</h4>{diff_text}</div>"

    yield base_box_html, tuned_box_html, diff_box_html, score_html

# =========================
# 5. Gradio UI
# =========================
with gr.Blocks(title="FLAN-T5 LoRA Ontario Demo") as demo:
    gr.Markdown("<h1 style='text-align:center;color:#336699;'>⚡ FLAN-T5 LoRA Demo</h1>"
                "<p style='text-align:center;'>Compare base vs LoRA-tuned predictions on Ontario electricity questions.</p>")

    with gr.Row():
        with gr.Column(scale=1):
            question_input = gr.Textbox(label="Type or select a question", lines=2)
            if example_questions:
                gr.Examples(examples=example_questions, inputs=question_input, label="Example Questions")
            show_scores_toggle = gr.Checkbox(value=True, label="Show Reference & Similarity Scores")
            compare_btn = gr.Button("🔍 Compare Predictions", variant="primary")

        with gr.Column(scale=2):
            base_box = gr.HTML(value=placeholder_card, label="Base Prediction")
            tuned_box = gr.HTML(value=placeholder_card, label="Tuned Prediction (LoRA)")
            diff_box = gr.HTML(value=placeholder_card, label="Differences Highlighted")
            score_box = gr.HTML(value=placeholder_card, label="Similarity / Reference")

    compare_btn.click(
        fn=compare_models,
        inputs=[question_input, show_scores_toggle],
        outputs=[base_box, tuned_box, diff_box, score_box]
    )

    gr.Markdown("<p style='text-align:center;opacity:0.6;'>Powered by 🤗 Transformers, PEFT, Gradio</p>")

demo.launch()
