import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================
# LOAD EMBEDDING MODEL
# ============================================
print("Loading multilingual embedding model for RAG...")
embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# ============================================
# LOAD ALL EMBEDDINGS
# ============================================
EMBED_FILES = {
    "english": "embeddings/english_embeddings.pkl",
    "marathi": "embeddings/marathi_embeddings.pkl",
    "hindi": "embeddings/hindi_embeddings.pkl",
    "bhojpuri": "embeddings/bhojpuri_embeddings.pkl"
}

all_data = []

for lang, path in EMBED_FILES.items():
    with open(path, "rb") as f:
        data = pickle.load(f)

    for text, label, emb in zip(data["texts"], data["labels"], data["embeddings"]):
        all_data.append({
            "text": text,
            "label": int(label),
            "embedding": emb
        })

label_map = {
    0: "Normal",
    1: "Hate",
    2: "Offensive"
}

def normalize_text(text):
    return str(text).strip().lower()

# ============================================
# LABEL-AWARE RAG RETRIEVAL
# ============================================
def multilingual_rag_retrieval(query_text, predicted_label=None, top_k=3, search_k=50):
    """
    query_text       : input sentence
    predicted_label  : model predicted class (0/1/2)
    top_k            : final results count
    search_k         : how many candidates to inspect
    """

    query_embedding = embed_model.encode([query_text], convert_to_numpy=True)[0]
    query_norm = normalize_text(query_text)

    # -------------------------------
    # Filter by predicted label if provided
    # -------------------------------
    if predicted_label is not None:
        candidate_data = [x for x in all_data if x["label"] == predicted_label]
    else:
        candidate_data = all_data

    candidate_texts = [x["text"] for x in candidate_data]
    candidate_labels = [x["label"] for x in candidate_data]
    candidate_embeddings = np.array([x["embedding"] for x in candidate_data])

    # Similarity
    scores = cosine_similarity([query_embedding], candidate_embeddings)[0]
    sorted_indices = np.argsort(scores)[::-1][:search_k]

    results = []
    seen_texts = set()

    for idx in sorted_indices:
        text = candidate_texts[idx]
        label = candidate_labels[idx]
        score = float(scores[idx])

        text_norm = normalize_text(text)

        # exact same text skip
        if text_norm == query_norm:
            continue

        # duplicate skip
        if text_norm in seen_texts:
            continue

        seen_texts.add(text_norm)

        results.append({
            "text": text,
            "label": label_map.get(label, str(label)),
            "score": score
        })

        if len(results) == top_k:
            break

    return results