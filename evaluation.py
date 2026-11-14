import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding
)

from torch.utils.data import Dataset, DataLoader

# =========================
# CONFIG
# =========================
DATA_PATH = "multilingual_dataset_safe_fixed.csv"
MODEL_PATH = "saved_model"

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATA_PATH)

df = df[["TEXT", "LABEL"]].dropna()
df["TEXT"] = df["TEXT"].astype(str)
df["LABEL"] = df["LABEL"].astype(int)

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["LABEL"]
)

test_df = test_df.reset_index(drop=True)

# =========================
# TOKENIZER
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

class HateDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts.tolist()
        self.labels = labels.tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        encoding = tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding=False,
            max_length=128,
            return_tensors="pt"
        )

        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx])

        return item


test_dataset = HateDataset(
    test_df["TEXT"],
    test_df["LABEL"]
)

collator = DataCollatorWithPadding(tokenizer)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    collate_fn=collator
)

# =========================
# LOAD TRAINED MODEL
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

model.to(device)
model.eval()

predictions = []
true_labels = []

# =========================
# EVALUATION
# =========================
with torch.no_grad():

    for batch in test_loader:

        labels = batch["labels"].cpu().numpy()

        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)

        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

        predictions.extend(preds)
        true_labels.extend(labels)

# =========================
# ACCURACY
# =========================
acc = accuracy_score(true_labels, predictions)

print("\nOverall Accuracy:")
print(acc)

# =========================
# CLASSIFICATION REPORT
# =========================
print("\nClassification Report:\n")

print(
    classification_report(
        true_labels,
        predictions,
        target_names=["Normal","Hate","Offensive"],
        digits=4
    )
)

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(true_labels, predictions)

print("\nConfusion Matrix:\n")
print(cm)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Normal","Hate","Offensive"],
    yticklabels=["Normal","Hate","Offensive"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig("confusion_matrix.png")

plt.show()

print("\nConfusion Matrix saved as confusion_matrix.png")