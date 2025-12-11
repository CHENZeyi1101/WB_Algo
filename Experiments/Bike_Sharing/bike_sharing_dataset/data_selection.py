import pandas as pd
import numpy as np

# Load the CSV
df = pd.read_csv("./WB_Algo/Experiments/Bike_Sharing/bike_sharing_dataset/hour.csv")

# Select desired columns
cols = ["cnt", "season", "hr", "workingday", "weathersit", "temp", "atemp", "hum", "windspeed"]
# Note: In the UCI Bike Sharing dataset, "hour" column is usually "hr"

selected = df[cols].copy()

# Add intercept column
selected["intercept"] = 1.0

selected.head()

# Shuffle the rows
selected = selected.sample(frac=1, random_state=42).reset_index(drop=True)

# save the selected columns to a new CSV
selected.to_csv("./WB_Algo/Experiments/Bike_Sharing/bike_sharing_selected_features.csv", index=False)

# y = "cnt" column (shape (n,))
y = selected["cnt"].to_numpy()

# X = all other columns (shape (n, 9))
X = selected.drop(columns=["cnt"]).to_numpy()

missing_summary = selected.isnull().sum()

print(missing_summary)

# Save to NPZ
np.savez("./WB_Algo/Experiments/Bike_Sharing/bike_sharing_data.npz", X=X, y=y)

print("Saved bike_sharing_data.npz")
print("X shape:", X.shape)
print("y shape:", y.shape)
