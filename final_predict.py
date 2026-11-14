import json
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util

# =========================
# CONFIG
# =========================
MODEL_PATH = "saved_model"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

LABEL_MAP = {
    0: "Normal",
    1: "Hate",
    2: "Offensive"
}

# =========================
# LOAD CLASSIFIER MODEL
# =========================
print("Loading classifier model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# =========================
# LOAD EMBEDDING MODEL
# =========================
print("Loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL)

# =========================
# LOAD KB FILES
# =========================
kb_files = [
    "kb/english_kb.json",
    "kb/hindi_kb.json",
    "kb/marathi_kb.json",
    "kb/bhojpuri_kb.json"
]

kb_data = []

print("\nLoading KB files...")
for file in kb_files:
    if not os.path.exists(file):
        print(f"[WARNING] File not found: {file}")
        continue

    try:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

            if isinstance(data, list):
                kb_data.extend(data)
                print(f"[OK] Loaded {file} -> {len(data)} rows")
            else:
                print(f"[WARNING] {file} does not contain a list")
    except Exception as e:
        print(f"[ERROR] Could not load {file}: {e}")

print(f"\nTotal KB rows loaded: {len(kb_data)}")

# =========================
# PREPARE KB TEXTS
# =========================
kb_texts = []
kb_labels = []

for item in kb_data:
    if isinstance(item, dict) and "TEXT" in item and "LABEL" in item:
        text = str(item["TEXT"]).strip()
        label = item["LABEL"]

        if text != "":
            kb_texts.append(text)
            kb_labels.append(int(label))

print(f"Valid KB texts found: {len(kb_texts)}")

# =========================
# CREATE KB EMBEDDINGS
# =========================
kb_embeddings = None

if len(kb_texts) > 0:
    print("Creating KB embeddings...")
    kb_embeddings = embedder.encode(kb_texts, convert_to_tensor=True, show_progress_bar=True)
    print("KB embeddings ready.")
else:
    print("[WARNING] No KB texts found. Retrieval will be disabled.")

# =========================
# PREDICT FUNCTION
# =========================
def predict_text(text, top_k=3):
    # ---------- CLASSIFIER ----------
    encoding = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )

    encoding = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_label_id = int(np.argmax(probs))
    pred_label = LABEL_MAP[pred_label_id]

    # ---------- RETRIEVAL ----------
    retrieved = []

    if kb_embeddings is not None and len(kb_texts) > 0:
        query_embedding = embedder.encode(text, convert_to_tensor=True)
        scores = util.cos_sim(query_embedding, kb_embeddings)[0]

        top_results = torch.topk(scores, k=min(top_k, len(kb_texts)))

        for score, idx in zip(top_results.values, top_results.indices):
            idx = int(idx)
            retrieved.append({
                "text": kb_texts[idx],
                "label": LABEL_MAP[kb_labels[idx]],
                "score": float(score)
            })

    return pred_label, probs, retrieved

# =========================
# INTERACTIVE LOOP
# =========================
print("\nModel is ready. Type text to test.")

while True:
    user_input = input("\nEnter text (or type 'exit' to stop): ").strip()

    if user_input.lower() == "exit":
        print("Exiting...")
        break

    if not user_input:
        print("Please enter some text.")
        continue

    pred_label, probs, retrieved = predict_text(user_input, top_k=3)

    print(f"\nPredicted Label: {pred_label}")
    print("Probabilities:")
    print(f"Normal     : {probs[0]:.4f}")
    print(f"Hate       : {probs[1]:.4f}")
    print(f"Offensive  : {probs[2]:.4f}")

    if len(retrieved) > 0:
        print("\nTop Retrieved Similar Examples:")
        for i, item in enumerate(retrieved, start=1):
            print(f"\nResult {i}")
            print("Text :", item["text"])
            print("Label:", item["label"])
            print("Score:", round(item["score"], 4))
    else:
        print("\nNo KB retrieval results available.")
        