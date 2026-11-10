
import pandas as pd
import numpy as np


np.random.seed(42)

data = []


pot_sizes = ["Small", "Medium", "Large"]
growing_media = ["Peat", "Sand", "Perlite"]
drainage_types = ["Poor", "Good"]


for plant_number in range(1, 31):

    plant_id = f"P{plant_number:02d}"

   
    pot_size = np.random.choice(pot_sizes)
    growing_medium = np.random.choice(growing_media)
    drainage = np.random.choice(drainage_types)

    for observation in range(10):

        
        target = np.random.choice([0, 1, 2])

        
        if target == 0:
            soil_moisture = np.random.normal(40, 8)
            time_since_watering = np.random.uniform(24, 96)
            watering_amount = np.random.uniform(50, 150)

        elif target == 1:
            soil_moisture = np.random.normal(65, 8)
            time_since_watering = np.random.uniform(12, 48)
            watering_amount = np.random.uniform(100, 250)

        else:
            soil_moisture = np.random.normal(85, 6)
            time_since_watering = np.random.uniform(1, 24)
            watering_amount = np.random.uniform(200, 400)

        
        soil_moisture = np.clip(soil_moisture, 10, 100)

        data.append({
            "plant_id": plant_id,
            "soil_moisture": round(soil_moisture, 2),
            "pot_size": pot_size,
            "growing_medium": growing_medium,
            "time_since_watering": round(time_since_watering, 2),
            "watering_amount": round(watering_amount, 2),
            "drainage": drainage,
            "target": target
        })


df = pd.DataFrame(data)


df.to_csv("overwatering_dataset.csv", index=False)


print("Dataset created successfully!")
print("\nFirst 10 observations:")
print(df.head(10))

print("\nDataset dimensions:")
print(df.shape)

print("\nNumber of observations in each class:")
print(df["target"].value_counts().sort_index())

print("\nDataset saved as overwatering_dataset.csv")
