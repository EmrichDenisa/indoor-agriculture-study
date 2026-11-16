
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

train_df = pd.read_csv("real_dataset/train_real.csv")
test_df = pd.read_csv("real_dataset/test_real.csv")

X_train = train_df.drop(columns=["target"])
y_train = train_df["target"]

X_test = test_df.drop(columns=["target"])
y_test = test_df["target"]

time_features = ["month", "day", "hour", "day_period", "quarter"]

experiments = {
    "All Features": X_train.columns.tolist(),
    "Without Time Features": [
        col for col in X_train.columns if col not in time_features
    ],
    "Without Month": [
        col for col in X_train.columns if col != "month"
    ]
}

results = []

for name, features in experiments.items():
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    model.fit(X_train[features], y_train)
    predictions = model.predict(X_test[features])

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Balanced Accuracy": balanced_accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1-score": f1_score(y_test, predictions, zero_division=0)
    })

    print("\n", name)
    print("Predicted classes:")
    print(pd.Series(predictions).value_counts().sort_index())
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

print("\nMODEL COMPARISON")
print(pd.DataFrame(results).to_string(index=False))
