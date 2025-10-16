#!/bin/bash

# Move to script directory
cd "$(dirname "$0")"

# Check if model exists
if [ ! -f "model/random_forest.h5" ]; then
    echo "Error: Model not found! Train the model first using ./train_and_predict.sh"
    exit 1
fi

# Run the prediction script with user inputs
python3 predict_model.py "$@"

# Notify user
echo "Prediction complete."