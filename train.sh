#!/bin/bash

echo "Starting Unified LTXV Training Pipeline..."

# Check if config file is provided
if [ $# -eq 0 ]; then
    echo "Usage: ./train.sh <config_file>"
    echo "Example: ./train.sh configs/unified_training_config.yaml"
    exit 1
fi

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Please activate your virtual environment first:"
    echo "source venv/bin/activate"
    exit 1
fi

# Run the training pipeline
python train.py "$1"