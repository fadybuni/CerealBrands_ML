#!/bin/bash

# Move to script directory
cd "$(dirname "$0")"

# Ensure model directory exists
mkdir -p model

# Run the model training script
python3 train_and_predict.py

# Notify user
echo "Model training complete. Model saved in model/random_forest.h5"