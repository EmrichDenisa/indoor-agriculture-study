
import pandas as pd

train_df = pd.read_csv("real_dataset/train_real.csv")
test_df = pd.read_csv("real_dataset/test_real.csv")

print("CLASS DISTRIBUTION BY MONTH")

df = pd.concat([train_df, test_df])
print(pd.crosstab(df["month"], df["target"]))

numeric_columns = [
    "weather_temp",
    "weather_humidity",
    "weather_rain",
    "weather_pressure",
    "weather_wind_speed",
    "weather_radiation",
    "soil_temperature_0-7cm",
    "soil_temperature_7-18cm",
    "ec",
    "ph",
    "soil_moisture",
    "irrigation_duration_minutes",
    "water_vol_past_4h"
]

print("\nTRAINING DATA - MEDIANS")
print(train_df[numeric_columns].median())

print("\nTESTING DATA - MEDIANS")
print(test_df[numeric_columns].median())

print("\nTARGET DISTRIBUTION")
print("Training:")
print(train_df["target"].value_counts(normalize=True))

print("\nTesting:")
print(test_df["target"].value_counts(normalize=True))
