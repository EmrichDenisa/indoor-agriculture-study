
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


df = pd.read_csv("overwatering_dataset.csv")


X = df.drop(columns=["target", "plant_id"])
y = df["target"]


groups = df["plant_id"]


splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_indices, test_indices = next(
    splitter.split(X, y, groups)
)

X_train = X.iloc[train_indices]
X_test = X.iloc[test_indices]

y_train = y.iloc[train_indices]
y_test = y.iloc[test_indices]


train_data = X_train.copy()
train_data["target"] = y_train

test_data = X_test.copy()
test_data["target"] = y_test

train_data.to_csv("train_dataset.csv", index=False)
test_data.to_csv("test_dataset.csv", index=False)


print("Training set size:", len(X_train))
print("Testing set size:", len(X_test))

print("\nTraining class distribution:")
print(y_train.value_counts().sort_index())

print("\nTesting class distribution:")
print(y_test.value_counts().sort_index())


train_plants = set(groups.iloc[train_indices])
test_plants = set(groups.iloc[test_indices])

print("\nPlants in training set:", len(train_plants))
print("Plants in testing set:", len(test_plants))

print("Common plants:", train_plants.intersection(test_plants))

print("\nDatasets prepared successfully!")
