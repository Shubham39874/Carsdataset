import pandas as pd
import xgboost as xg
import pickle
import os

# 1. Load Data
csv_path = os.path.join("model", "cars_24_combined.csv")
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()

# 2. Advanced Preprocessing
# Drop rows where Price is missing or zero
df = df[df['Price'] > 0].dropna(subset=['Price', 'Year', 'Distance', 'Fuel', 'Drive', 'Owner', 'Type', 'Location'])

# Mappings
fuel_map = {'Diesel': 1, 'Petrol': 2, 'CNG': 3, 'LPG': 4, 'Electric': 5}
drive_map = {'Manual': 1, 'Automatic': 2}
type_map = {'HatchBack': 1, 'Sedan': 2, 'SUV': 3, 'Lux_sedan': 4, 'Lux_SUV': 5}

df['Fuel'] = df['Fuel'].map(fuel_map)
df['Drive'] = df['Drive'].map(drive_map)
df['Type'] = df['Type'].map(type_map)

# Target Encoding for Location: Map each city to its mean price in this dataset
location_means = df.groupby('Location')['Price'].mean().to_dict()
df['Location_Score'] = df['Location'].map(location_means)

# Save the Location scores so app.py can use them
with open("models/location_scores.pkl", "wb") as f:
    pickle.dump(location_means, f)

# 3. Model Training
features = ['Year', 'Distance', 'Fuel', 'Drive', 'Owner', 'Type', 'Location_Score']
X = df[features]
y = df['Price']

# Exhaustive XGBoost Parameters
model = xg.XGBRegressor(
    n_estimators=500,        # More trees for exhaustive learning
    max_depth=6,             # Deeper trees to capture Location/Type nuances
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1
)

model.fit(X, y)

# 4. Save Assets
os.makedirs("models", exist_ok=True)
with open("models/car_pred", 'wb') as f:
    pickle.dump(model, f)

print(f"✅ Success! Trained on {len(df)} cars. Type & Location features included.")