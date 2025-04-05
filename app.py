# type: ignore
from flask import Flask, render_template, request, jsonify
import os
import tempfile
from werkzeug.utils import secure_filename
import torch
import whisper
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import difflib

# ----------------- SETUP FLASK APP -----------------
app = Flask(__name__)

# ----------------- LOAD MODELS -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"📦 Device: {device}")

print("🔊 Loading Whisper ASR model...")
asr_model = whisper.load_model("base")

print("📘 Loading Grammar Correction model...")
tokenizer = AutoTokenizer.from_pretrained("vennify/t5-base-grammar-correction")
grammar_model = AutoModelForSeq2SeqLM.from_pretrained("vennify/t5-base-grammar-correction").to(device)

# ----------------- UTIL FUNCTIONS -----------------
def transcribe_audio(asr_model, audio_path):
    result = asr_model.transcribe(audio_path)
    return result["text"]

def correct_grammar(model, tokenizer, text, device):
    input_text = "Correct the grammar: " + text.strip()
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(device)
    outputs = model.generate(**inputs, max_length=512)
    corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected

def compute_grammar_score(original, corrected):
    original_words = original.strip().split()
    corrected_words = corrected.strip().split()
    diff_count = sum(1 for o, c in zip(original_words, corrected_words) if o != c)
    diff_count += abs(len(original_words) - len(corrected_words))
    total = max(len(original_words), 1)
    score = max(0, 10 - int((diff_count / total) * 10))
    return score

def generate_diff_html(original, corrected):
    diff = difflib.ndiff(original.split(), corrected.split())
    html = []
    for word in diff:
        if word.startswith('-'):
            html.append(f'<span style="background:#fbb6b6;">{word[2:]}</span>')
        elif word.startswith('+'):
            html.append(f'<span style="background:#b6fbc4;">{word[2:]}</span>')
        elif word.startswith(' '):
            html.append(word[2:])
    return ' '.join(html)

# ----------------- ROUTES -----------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_audio():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio = request.files["audio"]
    filename = secure_filename(audio.filename)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        audio.save(tmp.name)
        audio_path = tmp.name

    transcribed_text = transcribe_audio(asr_model, audio_path)
    corrected_text = correct_grammar(grammar_model, tokenizer, transcribed_text, device)
    grammar_score = compute_grammar_score(transcribed_text, corrected_text)
    diff_html = generate_diff_html(transcribed_text, corrected_text)

    return jsonify({
        "transcribed": transcribed_text,
        "corrected": corrected_text,
        "score": grammar_score,
        "diff_html": diff_html
    })

# ----------------- MAIN -----------------
if __name__ == "__main__":
    app.run(debug=True)
