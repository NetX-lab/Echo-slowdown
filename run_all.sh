#!/bin/bash

# Run kernel_metric module
echo "Running kernel_metric module..."
cd kernel_metric
sh run.sh
if [ $? -ne 0 ]; then
    echo "Error: kernel_metric module failed."
    exit 1
fi
cd ..

# Run slowdown_collection module
echo "Running slowdown_collection module..."
cd slowdown_collection
sh run.sh
if [ $? -ne 0 ]; then
    echo "Error: slowdown_collection module failed."
    exit 1
fi
cd ..

# Run merge module
echo "Running merge module..."
cd merge
sh run.sh
if [ $? -ne 0 ]; then
    echo "Error: merge module failed."
    exit 1
fi
cd ..

# Run training_testing module
echo "Running training_testing module..."
cd training_testing
sh run.sh
if [ $? -ne 0 ]; then
    echo "Error: training_testing module failed."
    exit 1
fi
cd ..

echo "All modules ran successfully."