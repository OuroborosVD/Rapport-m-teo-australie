from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/weatherAUS.csv")
OUT_DIR = Path("outputs/splits_time")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
df = df.dropna(subset=["RainTomorrow"]).copy()
df["RainTomorrow_bin"] = df["RainTomorrow"].map({"No":0, "Yes":1})

# Trier par date
df = df.sort_values("Date").reset_index(drop=False)  # on garde l'index d'origine dans 'index'

n = len(df)
i1 = int(n*0.70)  # 70% train
i2 = int(n*0.85)  # +15% val, reste 15% test

train = df.iloc[:i1]
val   = df.iloc[i1:i2]
test  = df.iloc[i2:]

train[["Date","index"]].rename(columns={"index":"idx"}).to_csv(OUT_DIR/"train_idx.csv", index=False)
val[["Date","index"]].rename(columns={"index":"idx"}).to_csv(OUT_DIR/"val_idx.csv", index=False)
test[["Date","index"]].rename(columns={"index":"idx"}).to_csv(OUT_DIR/"test_idx.csv", index=False)

print("Dates:")
print("train:", train["Date"].min(), "→", train["Date"].max())
print("val  :", val["Date"].min(),   "→", val["Date"].max())
print("test :", test["Date"].min(),  "→", test["Date"].max())
