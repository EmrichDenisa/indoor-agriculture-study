
import pandas as pd

df = pd.read_csv("real_dataset/dataset_zone_3_preprocessed.csv")

df["ts"] = pd.to_datetime(df["ts"])
df = df.sort_values("ts").reset_index(drop=True)

df.loc[df["ec"] < 0, "ec"] = float("nan")
df.loc[(df["ph"] < 0) | (df["ph"] > 14), "ph"] = float("nan")

df["target"] = (df["target_point_24h"] >= 80).astype(int)

excluded_columns = [
    "ts",
    "target_point_24h",
    "target_mean_24h",
    "water_vol_to_24h",
    "real_moisture_delta"
]

df = df.drop(columns=excluded_columns)

split_index = int(len(df) * 0.8)

train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()

train_df.to_csv("real_dataset/train_real.csv", index=False)
test_df.to_csv("real_dataset/test_real.csv", index=False)

print("Dataset preparation completed!")

print("\nTraining observations:", len(train_df))
print("Testing observations:", len(test_df))

print("\nTraining class distribution:")
print(train_df["target"].value_counts().sort_index())

print("\nTesting class distribution:")
print(test_df["target"].value_counts().sort_index())

print("\nTraining period:")
print(train_df.index.min(), "-", split_index - 1)

print("\nTesting period:")
print(split_index, "-", len(df) - 1)

print("\nMissing values in training data:")
print(train_df.isnull().sum())

print("\nMissing values in testing data:")
print(test_df.isnull().sum())
