import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier

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



model = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("classifier", MLPClassifier(
        hidden_layer_sizes=(16, 8),
        activation="relu",
        max_iter=2000,
        random_state=42
    ))
])



model.fit(X_train, y_train)



y_pred = model.predict(X_test)



print("Neural Network - Soil Moisture Only")

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
