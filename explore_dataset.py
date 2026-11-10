
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("overwatering_dataset.csv")


print("First 10 observations:")
print(df.head(10))


print("\nDataset information:")
print(df.info())


print("\nNumerical feature statistics:")
print(df.describe())


print("\nMissing values:")
print(df.isnull().sum())


print("\nTarget class distribution:")
print(df["target"].value_counts().sort_index())


print("\nTarget class percentages:")
print((df["target"].value_counts(normalize=True).sort_index() * 100).round(2))


print("\nAverage feature values for each class:")
print(df.groupby("target")[
    ["soil_moisture", "time_since_watering", "watering_amount"]
].mean().round(2))


df["target"].value_counts().sort_index().plot(
    kind="bar",
    title="Distribution of Watering Risk Classes"
)

plt.xlabel("Target class")
plt.ylabel("Number of observations")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


df.boxplot(column="soil_moisture", by="target")

plt.title("Soil Moisture Distribution by Target Class")
plt.suptitle("")
plt.xlabel("Target class")
plt.ylabel("Soil moisture")
plt.tight_layout()
plt.show()
