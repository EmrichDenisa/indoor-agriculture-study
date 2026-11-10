
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


train_df = pd.read_csv("train_dataset.csv")
test_df = pd.read_csv("test_dataset.csv")


X_train = train_df[["soil_moisture"]]
y_train = train_df["target"]

X_test = test_df[["soil_moisture"]]
y_test = test_df["target"]


preprocessor = ColumnTransformer(
    transformers=[
        ("numerical", StandardScaler(), ["soil_moisture"])
    ]
)


model = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print("Logistic Regression - Soil Moisture Only")

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1, 2],
    target_names=["Normal", "At risk", "Overwatered"],
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
