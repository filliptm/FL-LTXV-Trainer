#!/bin/bash
echo "Setting up FL-LTXV-Trainer environment..."

# Check if virtual environment already exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Please create one first with:"
    echo "python -m venv venv"
    echo "source venv/bin/activate"
    echo "Then run this script again."
    exit 1
fi

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Please activate your virtual environment first:"
    echo "source venv/bin/activate"
    echo "Then run this script again."
    exit 1
fi

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Install package in development mode
echo "Installing FL-LTXV-Trainer in development mode..."
pip install -e .

echo ""
echo "Setup complete! The package is now installed in development mode."
echo "You can now run the preprocessing script or other tools."