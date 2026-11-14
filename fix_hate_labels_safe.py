import pandas as pd
import re

# =========================================================
# CONFIG
# =========================================================
INPUT_FILE = "multilingual_dataset.csv"
REVIEW_FILE = "suspicious_hate_rows.csv"
OUTPUT_FILE = "multilingual_dataset_safe_fixed.csv"

# =========================================================
# LOAD DATA
# =========================================================
df = pd.read_csv(INPUT_FILE)

print("Original shape:", df.shape)
print("\nOriginal label counts:")
print(df["LABEL"].value_counts())

# =========================================================
# CLEAN TEXT
# =========================================================
def normalize_text(text):
    if pd.isna(text):
        return ""
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text

df["TEXT"] = df["TEXT"].astype(str)
df["TEXT_CLEAN"] = df["TEXT"].apply(normalize_text)

# =========================================================
# TARGET GROUP WORDS
# Hate speech usually targets a group/community/caste/religion/etc.
# =========================================================
target_groups = [
    # Hindi
    "मुसलमान", "हिन्दू", "हिंदू", "दलित", "पाकिस्तानी", "जिहादी",
    "जात", "नीचली जात", "ऊँची जात", "समुदाय", "औरत", "महिला", "लोगों",

    # Marathi
    "मुस्लिम", "हिंदू", "दलित", "जिहादी", "समाज", "लोक", "जात", "धर्म",

    # Bhojpuri / mixed
    "लोग", "समुदाय", "जात", "धरम",

    # English
    "muslim", "hindu", "dalit", "community", "people", "women",
    "religion", "caste", "group", "terrorist"
]

# =========================================================
# STRONG HATE PATTERNS
# Only these kinds of expressions should push Normal -> Hate
# =========================================================
hate_patterns = [
    # Hindi
    r"नफरत",
    r"घृणा",
    r"मार\s*दूँगा",
    r"मार\s*डालो",
    r"खत्म\s*कर\s*दूँगा",
    r"देश\s*से\s*बाहर",
    r"इनको\s*मारो",
    r"इन\s*लोगों\s*को\s*मारो",
    r"इन\s*लोगों\s*को\s*हटा",
    r"इन\s*लोगों\s*को\s*निकालो",

    # Marathi
    r"तिरस्कार",
    r"द्वेष",
    r"मारून\s*टाका",
    r"संपवून\s*टाका",
    r"देशाबाहेर",
    r"हाकलून\s*द्या",
    r"नष्ट\s*करा",

    # Bhojpuri / mixed
    r"नफरत\s*बा",
    r"मार\s*देब",
    r"खतम\s*कर\s*देब",

    # English strong patterns only
    r"\bi hate these people\b",
    r"\bkill them\b",
    r"\bthrow them out\b",
    r"\bremove them\b",
    r"\bthey should die\b"
]

# =========================================================
# FILTERS TO AVOID FALSE POSITIVES
# These are not hate speech by themselves
# =========================================================
ignore_patterns = [
    r"i hate birds",
    r"i hate the yankees",
    r"killing two birds with one stone",
    r"killer line",
    r"killin it",
    r"trash talk",
    r"this drawing is trash",
    r"i hate flappy bird"
]

# =========================================================
# HELPERS
# =========================================================
def contains_target_group(text):
    return any(word in text for word in target_groups)

def matches_hate_pattern(text):
    for pattern in hate_patterns:
        if re.search(pattern, text):
            return True
    return False

def matches_ignore_pattern(text):
    for pattern in ignore_patterns:
        if re.search(pattern, text):
            return True
    return False

# =========================================================
# SUSPICIOUS NORMAL -> HATE RULE
# Only relabel if:
# 1) current label is Normal (0)
# 2) text has a target group/community word
# 3) text matches a strong hate pattern
# 4) text is not in ignore patterns
# =========================================================
def should_convert_normal_to_hate(text, label):
    if label != 0:
        return False

    if matches_ignore_pattern(text):
        return False

    if contains_target_group(text) and matches_hate_pattern(text):
        return True

    return False

# =========================================================
# FIND SUSPICIOUS ROWS
# =========================================================
suspicious_mask = df.apply(
    lambda row: should_convert_normal_to_hate(row["TEXT_CLEAN"], row["LABEL"]),
    axis=1
)

suspicious_rows = df[suspicious_mask].copy()

print("\n===================================================")
print("Suspicious rows for Normal -> Hate conversion:")
print("Count:", len(suspicious_rows))
print("===================================================\n")

if len(suspicious_rows) > 0:
    print(suspicious_rows[["TEXT", "LABEL"]].head(50).to_string(index=False))
else:
    print("No suspicious rows found.")

# Save for manual review
suspicious_rows[["TEXT", "LABEL"]].to_csv(REVIEW_FILE, index=False, encoding="utf-8")
print(f"\nReview file saved: {REVIEW_FILE}")

# =========================================================
# APPLY FIX
# =========================================================
df.loc[suspicious_mask, "LABEL"] = 1

# =========================================================
# DROP HELPER COLUMN
# =========================================================
df.drop(columns=["TEXT_CLEAN"], inplace=True)

# =========================================================
# SAVE FINAL FIXED DATASET
# =========================================================
df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

print("\n===================================================")
print("Safe fixed dataset saved:", OUTPUT_FILE)
print("===================================================\n")

print("Updated label counts:")
print(df["LABEL"].value_counts())