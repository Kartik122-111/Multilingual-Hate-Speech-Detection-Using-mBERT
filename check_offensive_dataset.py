import pandas as pd

df = pd.read_csv("multilingual_dataset_safe_fixed.csv")

print("Label Distribution:\n")
print(df["LABEL"].value_counts())

words = [
    "पागल",
    "झूठा",
    "मूर्ख",
    "बेवकूफ",
    "idiot",
    "stupid"
]

for word in words:
    print("\n" + "="*40)
    print("WORD:", word)

    result = df[
        (df["LABEL"] == 2) &
        (df["TEXT"].astype(str).str.contains(word, case=False, na=False))
    ]

    print("Rows Found:", len(result))

    if len(result):
        print(result[["TEXT", "LABEL"]].head(10).to_string(index=False))