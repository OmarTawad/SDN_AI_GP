#!/bin/bash

# Install system dependencies
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install -y python3.9 python3.9-venv

# Create and activate virtual environment
python3.9 -m venv ryu_env
source ryu_env/bin/activate

# Upgrade base tools
pip install --upgrade pip setuptools wheel

# Install Python dependencies from requirements.txt
pip install -r /MMLE/requirements.txt

# Optional: Confirm versions
echo "✔ Python version: $(python --version)"
echo "✔ Ryu version: $(pip show ryu | grep Version)"
