import pandas as pd
import numpy as np
import joblib
import h5py

import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load preprocessed data
data = pd.read_csv("/Users/alberthua/Desktop/ML_Project/output/preprocessed_data.csv")

# Assume last column is the target variable
X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]   # Target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train KNN model
model = KNeighborsClassifier(n_neighbors=5)  # You can tune n_neighbors
model.fit(X_train, y_train)

# Save model using .h5 format (storing separately)
model_filename = "/Users/alberthua/Desktop/ML_Project/model/knn_model.h5"
pkl_filename = "/Users/alberthua/Desktop/ML_Project/model/knn_model.pkl"

# Save the model using joblib
joblib.dump(model, pkl_filename)

# Save reference to the pkl file inside the h5 file
with h5py.File(model_filename, "w") as f:
    f.create_dataset("model_path", data=np.bytes_(pkl_filename))

print(f"KNN model trained and saved at {model_filename}")

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"KNN Model Accuracy: {accuracy:.2f}")
