import pandas as pd
from pathlib import Path

df = pd.read_csv("data/pairs_train.csv")

root = Path("data/processed")

missing = []

for col in ["img1", "img2"]:

    for path in df[col]:

        full_path = root / path

        if not full_path.exists():
            missing.append(str(full_path))

print("Missing:", len(missing))

for x in missing[:20]:
    print(x)