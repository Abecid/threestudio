#!/bin/bash

# Update system package list
echo "Updating system packages..."
sudo apt-get update

# Install system-wide dependencies
echo "Installing system-wide dependencies..."
# sudo apt-get install -y package1 package2

# Check if pip is installed, and install it if it isn't
# if ! command -v pip &> /dev/null
# then
#     echo "pip could not be found, installing..."
#     sudo apt-get install -y python3-pip
# fi

# Install Python dependencies
echo "Installing Python dependencies from requirements.txt..."
pip install -r requirements.txt

echo "Setup completed successfully!"
