
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


train_df = pd.read_csv("train_dataset.csv")
test_df = pd.read_csv("test_dataset.csv")

X_train_all = train_df.drop(columns=["target"])
y_train = train_df["target"]

X_test_all = test_df.drop(columns=["target"])
y_test = test_df["target"]

X_train_moisture = train_df[["soil_moisture"]]
X_test_moisture = test_df[["soil_moisture"]]

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

all_features_preprocessor = ColumnTransformer(
    transformers=[
        ("numerical", StandardScaler(), numerical_features),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

moisture_preprocessor = StandardScaler()

models = [
    (
        "Logistic Regression - Moisture Only",
        Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000))
        ]),
        X_train_moisture,
        X_test_moisture
    ),
    (
        "Logistic Regression - All Features",
        Pipeline([
            ("preprocessing", all_features_preprocessor),
            ("classifier", LogisticRegression(max_iter=1000))
        ]),
        X_train_all,
        X_test_all
    ),
    (
        "Random Forest - Moisture Only",
        Pipeline([
            ("classifier", RandomForestClassifier(
                n_estimators=100,
                random_state=42
            ))
        ]),
        X_train_moisture,
        X_test_moisture
    ),
    (
        "Random Forest - All Features",
        Pipeline([
            ("preprocessing", ColumnTransformer(
                transformers=[
                    ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features)
                ],
                remainder="passthrough"
            )),
            ("classifier", RandomForestClassifier(
                n_estimators=100,
                random_state=42
            ))
        ]),
        X_train_all,
        X_test_all
    ),
    (
        "Neural Network - Moisture Only",
        Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", MLPClassifier(
                hidden_layer_sizes=(16, 8),
                activation="relu",
                max_iter=2000,
                random_state=42
            ))
        ]),
        X_train_moisture,
        X_test_moisture
    ),
    (
        "Neural Network - All Features",
        Pipeline([
            ("preprocessing", all_features_preprocessor),
            ("classifier", MLPClassifier(
                hidden_layer_sizes=(16, 8),
                activation="relu",
                max_iter=2000,
                random_state=42
            ))
        ]),
        X_train_all,
        X_test_all
    )
]

results = []

for name, model, X_train, X_test in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "F1-score": f1_score(y_test, y_pred, average="macro", zero_division=0)
    })

results_df = pd.DataFrame(results)

print("\nModel Comparison:")
print(results_df.round(3).to_string(index=False))

results_df.to_csv("model_comparison.csv", index=False)

metrics = ["Accuracy", "Precision", "Recall", "F1-score"]

for metric in metrics:
    plt.figure(figsize=(12, 6))
    plt.barh(results_df["Model"], results_df[metric])
    plt.xlabel(metric)
    plt.title(f"Model Comparison - {metric}")
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(f"comparison_{metric.lower().replace('-', '_')}.png")
    plt.show()
