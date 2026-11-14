import os
import json
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# ============================================
# PATHS
# ============================================
KB_FILES = {
    "english": "kb/english_kb.json",
    "marathi": "kb/marathi_kb.json",
    "hindi": "kb/hindi_kb.json",
    "bhojpuri": "kb/bhojpuri_kb.json"
}

os.makedirs("embeddings", exist_ok=True)

# ============================================
# LOAD EMBEDDING MODEL
# ============================================
print("Loading multilingual embedding model...")
embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# ============================================
# CREATE EMBEDDINGS FOR EACH KB
# ============================================
for lang, kb_path in KB_FILES.items():
    print(f"\nCreating embeddings for {lang}...")

    with open(kb_path, "r", encoding="utf-8") as f:
        kb_data = json.load(f)

    texts = [item["text"] for item in kb_data]
    labels = [item["label"] for item in kb_data]

    embeddings = embed_model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    save_data = {
        "texts": texts,
        "labels": labels,
        "embeddings": embeddings
    }

    save_path = f"embeddings/{lang}_embeddings.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(save_data, f)

    print(f"{lang}_embeddings.pkl saved at: {save_path}")

print("\nAll embeddings created successfully.")