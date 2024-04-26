#!/bin/bash 

echo "Setting up environment"
set -x
set -u 
# if some command files exit with non-zero status, exit the script
set -e

python3 -m venv venv 
source venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
echo "Environment setup completed"