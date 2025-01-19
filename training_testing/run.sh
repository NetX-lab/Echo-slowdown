#!/bin/bash

rm -rf output
mkdir -p output
mkdir -p output/prediction

# Execute the Python scripts
echo "Running create_dataset.py..."
python create_dataset.py

echo "Running train.py..."
python train.py

echo "Running predict.py..."
python predict.py
