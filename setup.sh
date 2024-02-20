#!/bin/bash

# Add Environment Variables
# Add PATH and LD_LIBRARY_PATH environment variables (For nvcc; CUDA 12.2)
echo "Setting up CUDA environment variables..."
export PATH=/usr/local/cuda-12.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH

# Update system package list
echo "Updating system packages..."
apt-get update

# Install system-wide dependencies
echo "Installing system-wide dependencies..."
# apt-get install -y package1 package2

# Install Miniconda
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Check if pip is installed, and install it if it isn't
# if ! command -v pip &> /dev/null
# then
#     echo "pip could not be found, installing..."
#     sudo apt-get install -y python3-pip
# fi

# pip install virtualenv
# python3 -m virtualenv venv
# . venv/bin/activate

echo "Updating pip..."
python3 -m pip install --upgrade pip

# Install Python dependencies
echo "Installing Python dependencies from requirements.txt..."
# pip install torch torchvision
pip install ninja

# Default requirements
pip install -r requirements.txt

# zero123 requirements
pip install -r requirements-zero123.txt

# gradio requirements
pip install -r requirements-gradio.txt

echo "Setup completed successfully!"
