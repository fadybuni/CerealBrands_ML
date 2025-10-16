import sys
import numpy as np
import pandas as pd
import h5py
import joblib

# Load the trained model
model_filename = "/Users/alberthua/Desktop/ML_Project/model/knn_model.h5"

# Load the .pkl file path from the .h5 file
with h5py.File(model_filename, "r") as f:
    pkl_path = f["model_path"][()].decode("utf-8")  # Extract the saved model path

# Load the actual model from the .pkl file
model = joblib.load(pkl_path)

# Ensure the user provided input features
if len(sys.argv) < 2:
    print("Error: No input features provided.")
    print("Usage: ./predict.sh feature1 feature2 ... featureN")
    sys.exit(1)

# Convert input to a list of float numbers
input_values = [float(x) for x in sys.argv[1:]]

# Load feature names from the preprocessed dataset
data = pd.read_csv("/Users/alberthua/Desktop/ML_Project/output/preprocessed_data.csv")  # Load column names
feature_names = data.columns[:-1]  # Exclude the target column

# Ensure the input length matches the model's expected features
if len(input_values) != len(feature_names):
    print(f"Error: Expected {len(feature_names)} features, but got {len(input_values)}")
    sys.exit(1)

# Convert input to a DataFrame with correct column names
input_features = pd.DataFrame([input_values], columns=feature_names)

# Make prediction
prediction = model.predict(input_features)

# Display result
print(f"Predicted Class: {prediction[0]}")