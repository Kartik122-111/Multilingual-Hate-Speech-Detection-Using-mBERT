import pandas as pd

# =========================
# LOAD ALL 4 DATASETS
# =========================
English = pd.read_csv("data/English.csv")
Marathi = pd.read_excel("data/Marathi.xlsx")
Hindi = pd.read_excel("data/Hindi.xlsx")
Bhojpuri = pd.read_excel("data/Bhojpuri.xlsx")

# =========================
# RENAME COLUMNS
# =========================
English = English.rename(columns={"tweet": "TEXT", "class": "LABEL"})
Marathi = Marathi.rename(columns={"text": "TEXT", "label": "LABEL"})
Hindi = Hindi.rename(columns={"text": "TEXT", "label": "LABEL"})
Bhojpuri = Bhojpuri.rename(columns={"भोजपुरी_पाठ": "TEXT", "लेबल": "LABEL"})

English = English[["TEXT", "LABEL"]].copy()
Marathi = Marathi[["TEXT", "LABEL"]].copy()
Hindi = Hindi[["TEXT", "LABEL"]].copy()
Bhojpuri = Bhojpuri[["TEXT", "LABEL"]].copy()

# =========================
# LABEL MAPPING
# =========================

# English
English["LABEL"] = English["LABEL"].replace({
    2: 0,
    0: 1,
    1: 2
})

# Marathi
Marathi["LABEL"] = Marathi["LABEL"].astype(str).str.strip().str.lower()
marathi_mapping = {
    "normal": 0,
    "not": 0,
    "non-hate": 0,
    "hate": 1,
    "offensive": 2,
    "offn": 2,
    "prfn": 2
}
Marathi["LABEL"] = Marathi["LABEL"].replace(marathi_mapping)

# Hindi
Hindi["LABEL"] = Hindi["LABEL"].astype(str).str.strip().str.lower()

def map_hindi_label(label):
    if label in ["normal", "non-hate", "non hostile", "non-hostile", "safe", "neither"]:
        return 0
    if label in ["hate", "hate speech"]:
        return 1
    if label in ["offensive", "offensive language", "vulgar", "violence", "defamation", "fake", "abusive"]:
        return 2
    if "hate" in str(label):
        return 1
    offensive_words = ["offensive", "vulgar", "violence", "defamation", "fake", "abuse"]
    if any(word in str(label) for word in offensive_words):
        return 2
    return label

Hindi["LABEL"] = Hindi["LABEL"].apply(map_hindi_label)

# Bhojpuri
Bhojpuri["LABEL"] = Bhojpuri["LABEL"].astype(int)

# =========================
# CLEAN
# =========================
English = English[English["LABEL"].isin([0, 1, 2])]
Marathi = Marathi[Marathi["LABEL"].isin([0, 1, 2])]
Hindi = Hindi[Hindi["LABEL"].isin([0, 1, 2])]
Bhojpuri = Bhojpuri[Bhojpuri["LABEL"].isin([0, 1, 2])]

for df in [English, Marathi, Hindi, Bhojpuri]:
    df["TEXT"] = df["TEXT"].astype(str).str.strip()

English = English[English["TEXT"] != ""]
Marathi = Marathi[Marathi["TEXT"] != ""]
Hindi = Hindi[Hindi["TEXT"] != ""]
Bhojpuri = Bhojpuri[Bhojpuri["TEXT"] != ""]

# =========================
# MERGE
# =========================
mg = pd.concat([English, Marathi, Hindi, Bhojpuri], ignore_index=True)
mg = mg.dropna(subset=["TEXT", "LABEL"])
mg["LABEL"] = mg["LABEL"].astype(int)

mg.to_csv("multilingual_dataset.csv", index=False, encoding="utf-8")

print("Merged dataset saved as multilingual_dataset.csv")
print("Shape:", mg.shape)
print(mg["LABEL"].value_counts())