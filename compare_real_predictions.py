
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

train_df = pd.read_csv("real_dataset/train_real.csv")
test_df = pd.read_csv("real_dataset/test_real.csv")

X_train = train_df.drop(columns=["target"])
y_train = train_df["target"]

X_test = test_df.drop(columns=["target"])
y_test = test_df["target"]

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

train_probabilities = model.predict_proba(X_train)[:, 1]
test_probabilities = model.predict_proba(X_test)[:, 1]

train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

print("TRAINING PERFORMANCE")
print(classification_report(
    y_train,
    train_predictions,
    target_names=["No Excess Risk", "Excess Risk"],
    zero_division=0
))
print("Confusion Matrix:")
print(confusion_matrix(y_train, train_predictions))

print("\nTESTING PERFORMANCE")
print(classification_report(
    y_test,
    test_predictions,
    target_names=["No Excess Risk", "Excess Risk"],
    zero_division=0
))
print("Confusion Matrix:")
print(confusion_matrix(y_test, test_predictions))

print("\nMEAN RISK PROBABILITY BY ACTUAL CLASS")

print("Training:")
print(pd.DataFrame({
    "actual": y_train,
    "probability": train_probabilities
}).groupby("actual")["probability"].describe())

print("\nTesting:")
print(pd.DataFrame({
    "actual": y_test,
    "probability": test_probabilities
}).groupby("actual")["probability"].describe())

print("\nLOGISTIC REGRESSION COEFFICIENTS")

coefficients = pd.DataFrame({
    "Feature": X_train.columns,
    "Coefficient": model.named_steps["classifier"].coef_[0]
})

print(coefficients.sort_values(
    by="Coefficient",
    key=abs,
    ascending=False
).to_string(index=False))
