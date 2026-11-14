
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("overwatering_dataset.csv")

groups = (
    df["pot_size"].astype(str) + "_" +
    df["growing_medium"].astype(str)
)

X = df.drop(columns=["target", "plant_id"])
y = df["target"]

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=42
)

train_indices, test_indices = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_indices]
X_test = X.iloc[test_indices]
y_train = y.iloc[train_indices]
y_test = y.iloc[test_indices]

train_groups = groups.iloc[train_indices]
test_groups = groups.iloc[test_indices]

print("Training observations:", len(X_train))
print("Testing observations:", len(X_test))

print("\nTraining combinations:")
print(sorted(train_groups.unique()))

print("\nTesting combinations:")
print(sorted(test_groups.unique()))

print("\nCommon combinations:")
print(set(train_groups.unique()) & set(test_groups.unique()))

print("\nTraining class distribution:")
print(y_train.value_counts().sort_index())

print("\nTesting class distribution:")
print(y_test.value_counts().sort_index())

numerical_features = [
    "soil_moisture",
    "time_since_watering",
    "watering_amount"
]

categorical_features = [
    "pot_size",
    "growing_medium",
    "drainage"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("numerical", StandardScaler(), numerical_features),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

models = {
    "Logistic Regression - Moisture Only": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000))
    ]),
    "Logistic Regression - All Features": Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]),
    "Random Forest - Moisture Only": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),
    "Random Forest - All Features": Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
    ]),
    "Neural Network - Moisture Only": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", MLPClassifier(
            hidden_layer_sizes=(16, 8),
            activation="relu",
            max_iter=2000,
            random_state=42
        ))
    ]),
    "Neural Network - All Features": Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", MLPClassifier(
            hidden_layer_sizes=(16, 8),
            activation="relu",
            max_iter=2000,
            random_state=42
        ))
    ])
}

results = []

for name, model in models.items():
    if "Moisture Only" in name:
        model.fit(X_train[["soil_moisture"]], y_train)
        predictions = model.predict(X_test[["soil_moisture"]])
    else:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, average="macro", zero_division=0),
        "Recall": recall_score(y_test, predictions, average="macro", zero_division=0),
        "F1-score": f1_score(y_test, predictions, average="macro", zero_division=0)
    })

results_df = pd.DataFrame(results)

print("\nUnseen Combinations Model Comparison:")
print(results_df.to_string(index=False))

results_df.to_csv("unseen_combinations_comparison.csv", index=False)
