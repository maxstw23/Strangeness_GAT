#!/bin/bash
set -e

PYTHON_VERSION="3.10"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== OmegaNet Setup Script ==="
echo "Project root: $PROJECT_ROOT"
echo "Python version: $PYTHON_VERSION"
echo ""

# Check if Python is installed
if ! command -v python$PYTHON_VERSION &> /dev/null; then
    echo "❌ Python $PYTHON_VERSION not found"
    echo ""
    echo "Install Python $PYTHON_VERSION using your system package manager:"
    echo "  Ubuntu/Debian: sudo apt install -y python$PYTHON_VERSION python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-dev"
    echo "  macOS: brew install python@${PYTHON_VERSION}"
    echo "  Other: Visit https://www.python.org/downloads/"
    exit 1
fi

echo "✓ Python $PYTHON_VERSION found: $(python$PYTHON_VERSION --version)"
echo ""

# Remove old venv if it exists
if [ -d "$PROJECT_ROOT/venv" ]; then
    echo "Removing old venv..."
    rm -rf "$PROJECT_ROOT/venv"
fi

# Create venv
echo "Creating virtual environment..."
python$PYTHON_VERSION -m venv "$PROJECT_ROOT/venv"

# Activate venv
echo "Activating virtual environment..."
source "$PROJECT_ROOT/venv/bin/activate"

# Install PyTorch with CUDA 12.4 support
echo "Installing PyTorch with CUDA 12.4..."
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Install tqdm
echo "Installing tqdm..."
pip install tqdm

# Install uproot and awkward (for ROOT file handling)
echo "Installing uproot and awkward..."
pip install uproot awkward

# Install scipy, matplotlib, and iminuit (for efficiency fitting and plotting)
echo "Installing scipy, matplotlib, and iminuit..."
pip install scipy matplotlib iminuit

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To activate the venv in future sessions, run:"
echo "  source venv/bin/activate"
echo ""
echo "To train, run:"
echo "  source venv/bin/activate"
echo "  python scripts/train.py"
