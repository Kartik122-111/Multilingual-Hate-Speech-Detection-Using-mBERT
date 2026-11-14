import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm

# =========================
# CONFIG
# =========================
DATA_PATH = "multilingual_dataset_safe_fixed.csv"
MODEL_NAME = "bert-base-multilingual-cased"
SAVE_DIR = "saved_model"
MAX_LEN = 128
BATCH_SIZE = 8
LR = 2e-5

# If saved_model already exists and contains 1 trained epoch,
# then this will run remaining 2 epochs.
EPOCHS = 3

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATA_PATH)

# keep only required columns
df = df[["TEXT", "LABEL"]].dropna()
df["TEXT"] = df["TEXT"].astype(str)
df["LABEL"] = df["LABEL"].astype(int)

print("Dataset shape:", df.shape)
print("\nLabel distribution:")
print(df["LABEL"].value_counts())

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["LABEL"]
)

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# =========================
# TOKENIZER
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# =========================
# DATASET CLASS
# =========================
class HateDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=False,
            max_length=self.max_len,
            return_tensors="pt"
        )

        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item

train_dataset = HateDataset(train_df["TEXT"], train_df["LABEL"], tokenizer, MAX_LEN)
test_dataset = HateDataset(test_df["TEXT"], test_df["LABEL"], tokenizer, MAX_LEN)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=data_collator
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=data_collator
)

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\nUsing device:", device)

# =========================
# LOAD MODEL (RESUME IF EXISTS)
# =========================
if os.path.exists(SAVE_DIR):
    print(f"\nLoading existing trained model from '{SAVE_DIR}' ...")
    model = AutoModelForSequenceClassification.from_pretrained(SAVE_DIR)
else:
    print("\nNo saved model found. Loading fresh base model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3
    )

model.to(device)

# =========================
# OPTIMIZER
# =========================
optimizer = AdamW(model.parameters(), lr=LR)

# =========================
# TRAINING
# =========================
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for batch in loop:
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)
    print(f"\nEpoch {epoch+1} Average Loss: {avg_loss:.4f}")

# =========================
# EVALUATION
# =========================
model.eval()
predictions = []
true_labels = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating"):
        labels = batch["labels"].cpu().numpy()
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

        predictions.extend(preds)
        true_labels.extend(labels)

acc = accuracy_score(true_labels, predictions)
print("\nAccuracy:", acc)

print("\nClassification Report:")
print(classification_report(true_labels, predictions, digits=4))

# =========================
# SAVE MODEL
# =========================
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

print(f"\nModel saved in folder: {SAVE_DIR}")
