import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("Data/train.csv")  # Change path if needed

# Encode categorical features
df = pd.get_dummies(df, drop_first=True)

# Features and Target
X = df.drop(['price'], axis=1)
y = df['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model to a folder
os.makedirs("Model", exist_ok=True)
joblib.dump(model, "Model/house_price_model.pkl")
# Save feature names
joblib.dump(X.columns.tolist(), "Model/features.pkl")

# Evaluate
y_pred = model.predict(X_test)
print("✅ Evaluation:")
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))
