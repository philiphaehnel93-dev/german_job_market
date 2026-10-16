#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e 

echo "🚀 Bootstrapping ML Pipeline Environment..."

# 1. Check if uv is installed (fails gracefully if missing)
if ! command -v uv &> /dev/null; then
    echo "Error: 'uv' is not installed. Please run:"
    echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# 2. Create the virtual environment
echo "📦 Creating virtual environment..."
uv venv

# 3. Install dependencies 
# (You can replace this with 'uv pip install -r requirements.txt' later)
echo "📥 Installing core dependencies..."
uv pip install google-cloud-storage pandas scikit-learn

echo "✅ Setup complete! Run 'source .venv/bin/activate' to start working."
