import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import pickle
import re

# Load dataset
df = pd.read_csv("car_data.csv")

# Display column names to verify
print("Columns in dataset:", df.columns)

# Remove extra spaces (if any) from column names
df.columns = df.columns.str.strip()

# Clean 'kms_driven' column
df["kms_driven"] = df["kms_driven"].astype(str).apply(lambda x: re.sub(r"[^0-9]", "", x))  # Remove non-numeric characters
df["kms_driven"] = pd.to_numeric(df["kms_driven"], errors='coerce')  # Convert to float

# Clean 'Price' column (remove commas and convert to float)
df["Price"] = df["Price"].astype(str).apply(lambda x: re.sub(r"[^0-9]", "", x))  # Remove non-numeric characters
df["Price"] = pd.to_numeric(df["Price"], errors='coerce')  # Convert to float

# Drop rows where 'kms_driven' or 'Price' is still NaN after cleaning
df = df.dropna(subset=["kms_driven", "Price"])

# Encoding categorical variables
label_encoders = {}
for col in ["company", "fuel_type"]:  # Encoding only company and fuel_type
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Selecting features and target variable
X = df[['company', 'year', 'kms_driven', 'fuel_type']]
y = df['Price']

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
with open("car_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save label encoders
with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

print("Model training completed and saved!")
