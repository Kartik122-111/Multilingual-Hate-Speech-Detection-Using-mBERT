import os
import json
import pandas as pd

# ============================================
# FILE PATHS
# ============================================
FILES = {
    "english": r"data\English.csv",
    "marathi": r"data\Marathi.xlsx",
    "hindi": r"data\Hindi.xlsx",
    "bhojpuri": r"data\Bhojpuri.xlsx"
}

os.makedirs("kb", exist_ok=True)

# ============================================
# LABEL MAPPING
# Final labels:
# 0 = Normal
# 1 = Hate
# 2 = Offensive
# ============================================

def map_english_label(x):
    # English original:
    # 0 = Hate Speech
    # 1 = Offensive Language
    # 2 = Neither
    try:
        x = int(x)
    except:
        return x

    if x == 2:
        return 0   # Neither -> Normal
    elif x == 0:
        return 1   # Hate Speech -> Hate
    elif x == 1:
        return 2   # Offensive Language -> Offensive
    return x


def map_marathi_label(x):
    x = str(x).strip().lower()

    mapping = {
        # normal
        "not": 0,
        "normal": 0,
        "normal speech": 0,
        "non-hate": 0,
        "non hostile": 0,
        "non-hostile": 0,
        "सामान्य": 0,

        # hate
        "hate": 1,
        "hate speech": 1,
        "द्वेष": 1,

        # offensive
        "offn": 2,
        "prfn": 2,
        "offensive": 2,
        "offensive language": 2,
        "abusive": 2,
        "आक्षेपार्ह": 2,
        "अश्लील": 2,
        "अपमानजनक": 2
    }

    return mapping.get(x, x)


def map_hindi_label(x):
    x = str(x).strip().lower()

    # normal
    if x in ["normal", "non-hate", "non hostile", "non-hostile", "safe", "neither", "सामान्य"]:
        return 0

    # hate
    if x in ["hate", "hate speech", "द्वेष"]:
        return 1

    # offensive
    if x in [
        "offensive", "offensive language", "vulgar", "violence",
        "defamation", "fake", "abusive", "आक्षेपार्ह", "अश्लील"
    ]:
        return 2

    if "hate" in x:
        return 1

    offensive_words = ["offensive", "vulgar", "violence", "defamation", "fake", "abuse"]
    if any(word in x for word in offensive_words):
        return 2

    return x


def map_bhojpuri_label(x):
    x = str(x).strip().lower()

    mapping = {
        "0": 0,
        "1": 1,
        "2": 2,
        "normal": 0,
        "hate": 1,
        "offensive": 2,
        "सामान्य": 0,
        "द्वेष": 1,
        "आक्षेपार्ह": 2
    }

    return mapping.get(x, x)


# ============================================
# FILE LOADER
# ============================================
def load_file(path):
    ext = os.path.splitext(path)[1].lower()

    if ext == ".csv":
        try:
            return pd.read_csv(path)
        except:
            try:
                return pd.read_csv(path, encoding="latin1")
            except:
                return pd.read_csv(path, encoding="utf-8", engine="python", on_bad_lines="skip")

    elif ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)

    else:
        raise ValueError(f"Unsupported file format: {path}")


# ============================================
# DETECT TEXT / LABEL COLUMNS
# ============================================
def detect_text_label_columns(df, lang):
    cols = list(df.columns)

    text_col = None
    label_col = None

    # ---------------- ENGLISH ----------------
    if lang == "english":
        if "tweet" in cols:
            text_col = "tweet"
        elif "text" in cols:
            text_col = "text"
        elif "TEXT" in cols:
            text_col = "TEXT"

        if "class" in cols:
            label_col = "class"
        elif "label" in cols:
            label_col = "label"
        elif "LABEL" in cols:
            label_col = "LABEL"

    # ---------------- MARATHI / HINDI ----------------
    elif lang in ["marathi", "hindi"]:
        for c in cols:
            cl = str(c).strip().lower()

            if cl in ["text", "tweet", "sentence", "comment", "content", "post"]:
                text_col = c

            if cl in ["label", "class", "category", "target"]:
                label_col = c

    # ---------------- BHOJPURI ----------------
    elif lang == "bhojpuri":
        # Expected columns:
        # ['आईडी', 'भोजपुरी_पाठ', 'श्रेणी', 'लेबल', 'भावना']
        for c in cols:
            if str(c).strip() == "भोजपुरी_पाठ":
                text_col = c
            if str(c).strip() == "लेबल":
                label_col = c

    # ---------------- FALLBACK ----------------
    if text_col is None:
        text_col = cols[0]

    if label_col is None:
        if len(cols) > 1:
            label_col = cols[1]
        else:
            raise ValueError(f"Could not detect label column in {lang} dataset")

    return text_col, label_col


# ============================================
# PROCESS EACH DATASET
# ============================================
for lang, path in FILES.items():
    print(f"\n{'='*60}")
    print(f"Processing {lang}...")
    print(f"File: {path}")

    df = load_file(path)

    print("Original columns:", list(df.columns))
    print("Original shape:", df.shape)

    text_col, label_col = detect_text_label_columns(df, lang)
    print(f"Detected TEXT column: {text_col}")
    print(f"Detected LABEL column: {label_col}")

    # Rename to common format
    df = df.rename(columns={text_col: "TEXT", label_col: "LABEL"})

    # Keep only needed columns
    df = df[["TEXT", "LABEL"]].copy()

    # ============================================
    # LABEL MAPPING
    # ============================================
    if lang == "english":
        df["LABEL"] = df["LABEL"].apply(map_english_label)

    elif lang == "marathi":
        df["LABEL"] = df["LABEL"].apply(map_marathi_label)

    elif lang == "hindi":
        df["LABEL"] = df["LABEL"].apply(map_hindi_label)

    elif lang == "bhojpuri":
        df["LABEL"] = df["LABEL"].apply(map_bhojpuri_label)

    # ============================================
    # CLEAN
    # ============================================
    df["TEXT"] = df["TEXT"].astype(str).str.strip()
    df["LABEL"] = pd.to_numeric(df["LABEL"], errors="coerce")

    df = df.dropna(subset=["TEXT", "LABEL"])
    df = df[df["TEXT"] != ""]
    df = df[df["LABEL"].isin([0, 1, 2])]
    df["LABEL"] = df["LABEL"].astype(int)

    print("\nCleaned shape:", df.shape)
    print("Label counts:")
    print(df["LABEL"].value_counts())

    print(f"\nSample rows from {lang}:")
    print(df[["TEXT", "LABEL"]].head(10))

    print("Unique labels:", df["LABEL"].unique())

    # ============================================
    # SAVE KB JSON
    # ============================================
    kb_data = []
    for _, row in df.iterrows():
        kb_data.append({
            "text": row["TEXT"],
            "label": int(row["LABEL"])
        })

    save_path = os.path.join("kb", f"{lang}_kb.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(kb_data, f, ensure_ascii=False, indent=2)

    print(f"\nSaved: {save_path}")

print("\nAll KB files created successfully.")