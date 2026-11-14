import pandas as pd

df = pd.read_csv("multilingual_dataset_safe_fixed.csv")

words = [
    "idiot",
    "stupid",
    "dumb",
    "moron",
    "loser",
    "fool",
    "dirty"
]

for word in words:
    print("\n==============================")
    print(word.upper())
    print("==============================")

    temp = df[df["TEXT"].str.contains(word, case=False, na=False)]

    if len(temp) == 0:
        print("No rows found.")
    else:
        print(temp[["TEXT","LABEL"]].head(30).to_string(index=False))