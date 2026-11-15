
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

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

probabilities = model.predict_proba(X_test)[:, 1]
predictions = model.predict(X_test)

results = pd.DataFrame({
    "month": test_df["month"],
    "actual": y_test,
    "predicted": predictions,
    "risk_probability": probabilities
})

print("PREDICTED CLASS DISTRIBUTION")
print(results["predicted"].value_counts().sort_index())

print("\nPREDICTED PROBABILITY STATISTICS BY ACTUAL CLASS")
print(results.groupby("actual")["risk_probability"].describe())

print("\nPREDICTIONS BY MONTH")
print(pd.crosstab(
    results["month"],
    results["predicted"],
    rownames=["Month"],
    colnames=["Predicted class"]
))

print("\nEXAMPLES OF PREDICTIONS")
print(results.head(20).to_string(index=False))
