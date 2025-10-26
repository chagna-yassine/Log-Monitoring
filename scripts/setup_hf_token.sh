#!/bin/bash

echo "========================================"
echo "HUGGING FACE TOKEN SETUP (Linux/Mac)"
echo "========================================"
echo
echo "This script will help you set up your Hugging Face token."
echo
echo "1. Go to: https://huggingface.co/settings/tokens"
echo "2. Click 'New token'"
echo "3. Give it a name (e.g., 'ait-processing')"
echo "4. Select 'Write' permissions"
echo "5. Copy the token"
echo

read -p "Enter your Hugging Face token: " HF_TOKEN

if [ -z "$HF_TOKEN" ]; then
    echo "Error: No token provided."
    exit 1
fi

echo
echo "Setting token for current session..."
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"

echo
echo "Testing token..."
python3 -c "import os; from huggingface_hub import HfApi; api = HfApi(); print('✅ Token works! Authenticated as:', api.whoami()['name'])"

if [ $? -eq 0 ]; then
    echo
    echo "✅ SUCCESS! Your token is working."
    echo
    echo "To make this permanent, add this line to your ~/.bashrc or ~/.zshrc:"
    echo "export HUGGINGFACE_HUB_TOKEN=\"$HF_TOKEN\""
    echo
    echo "Or run this command:"
    echo "echo 'export HUGGINGFACE_HUB_TOKEN=\"$HF_TOKEN\"' >> ~/.bashrc"
else
    echo
    echo "❌ Token test failed. Please check your token and try again."
fi

echo
read -p "Press Enter to continue..."
