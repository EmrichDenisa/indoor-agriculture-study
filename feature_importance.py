
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

train_df = pd.read_csv("train_dataset.csv")
test_df = pd.read_csv("test_dataset.csv")

X_train = train_df.drop(columns=["target"])
y_train = train_df["target"]

X_test = test_df.drop(columns=["target"])
y_test = test_df["target"]

categorical_features = ["pot_size", "growing_medium", "drainage"]
numerical_features = ["soil_moisture", "time_since_watering", "watering_amount"]

preprocessor = ColumnTransformer(
    transformers=[
        ("numerical", "passthrough", numerical_features),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

model.fit(X_train, y_train)

results = permutation_importance(
    model,
    X_test,
    y_test,
    scoring="f1_macro",
    n_repeats=20,
    random_state=42
)

importance_df = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance": results.importances_mean,
    "Standard Deviation": results.importances_std
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("Feature Importance:")
print(importance_df.to_string(index=False))

importance_df.to_csv("feature_importance.csv", index=False)

plt.figure(figsize=(10, 6))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.gca().invert_yaxis()
plt.xlabel("Decrease in Macro F1-score")
plt.title("Permutation Feature Importance")
plt.tight_layout()
plt.savefig("feature_importance.png") 
plt.show()
