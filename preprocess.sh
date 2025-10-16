#!/bin/bash
cd "$(dirname "$0")"
mkdir -p output
python3 data_preprocessing.py
echo "Data preprocessing complete. Processed dataset saved in output/preprocessed_data.csv"