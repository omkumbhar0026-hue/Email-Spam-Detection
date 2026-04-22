import pandas as pd

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

print("=== TRAIN ===")
print("Columns:", train.columns.tolist())
print("Shape:", train.shape)
print("NaN counts:\n", train.isnull().sum())
print(train.head(3))

print("\n=== TEST ===")
print("Columns:", test.columns.tolist())
print("Shape:", test.shape)
print("NaN counts:\n", test.isnull().sum())
print(test.head(3))
