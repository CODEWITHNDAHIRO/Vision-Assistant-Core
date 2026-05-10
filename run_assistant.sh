#!/bin/bash

# 1. Print a professional startup message
echo "------------------------------------------"
echo "Initializing Vision Assistant Core v2.1..."
echo "------------------------------------------"

# 2. Activate the specific environment
# Note: On Mac, we source the conda profile first
source ~/anaconda3/etc/profile.d/conda.sh
conda activate vision-env

# 3. Launch the Python brain
python assistant.py

# 4. Handle closure
echo "System Shutting Down Safely."
