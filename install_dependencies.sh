#!/bin/bash

# Hybrid Book Recommender System - Bash Installation Script
# Run this script from the project root directory

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  Hybrid Book Recommender System Setup${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
}

print_error() {
    echo -e "${RED}ERROR: $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}Warning: $1${NC}"
}

print_info() {
    echo -e "${CYAN}$1${NC}"
}

# Check if running on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    IS_MACOS=true
else
    IS_MACOS=false
fi

print_status

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed or not in PATH"
    echo "Please install Python 3.8 or higher:"
    if [ "$IS_MACOS" = true ]; then
        echo "  brew install python3"
        echo "  or download from https://python.org"
    else
        echo "  sudo apt-get install python3 python3-pip  # Ubuntu/Debian"
        echo "  sudo yum install python3 python3-pip      # CentOS/RHEL"
        echo "  or download from https://python.org"
    fi
    exit 1
fi

print_success "Python is installed"
python3 --version

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    print_error "pip3 is not available"
    echo "Please install pip3:"
    if [ "$IS_MACOS" = true ]; then
        echo "  brew install python3"
    else
        echo "  sudo apt-get install python3-pip  # Ubuntu/Debian"
        echo "  sudo yum install python3-pip      # CentOS/RHEL"
    fi
    exit 1
fi

print_success "pip3 is available"

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    print_error "requirements.txt not found in current directory"
    echo "Please run this script from the project root directory"
    exit 1
fi

# Upgrade pip to latest version
print_info "Upgrading pip to latest version..."
if python3 -m pip install --upgrade pip; then
    print_success "pip upgraded successfully"
else
    print_warning "Failed to upgrade pip, continuing..."
fi

# Install setuptools and wheel for better package installation
print_info "Installing setuptools and wheel..."
if pip3 install setuptools wheel; then
    print_success "setuptools and wheel installed"
else
    print_warning "Failed to install setuptools/wheel, continuing..."
fi

# Install project dependencies
echo ""
print_info "Installing project dependencies..."
print_info "This may take a few minutes..."

if pip3 install -r requirements.txt; then
    print_success "All dependencies installed successfully!"
else
    echo ""
    print_error "Failed to install some dependencies"
    echo "Please check the error messages above"
    echo ""
    echo "Troubleshooting tips:"
    echo "1. Try running with sudo (if permission issues)"
    echo "2. Install system dependencies:"
    if [ "$IS_MACOS" = true ]; then
        echo "   brew install gcc"
    else
        echo "   sudo apt-get install build-essential python3-dev  # Ubuntu/Debian"
        echo "   sudo yum groupinstall 'Development Tools'         # CentOS/RHEL"
    fi
    echo "3. Try: pip3 install --upgrade pip setuptools wheel"
    echo "4. Check your internet connection"
    exit 1
fi

# Verify installation
echo ""
print_info "Verifying installation..."
if python3 -c "import streamlit, pandas, numpy, sklearn, plotly, joblib; print('✓ All packages imported successfully')"; then
    print_success "Installation verified!"
else
    print_warning "Some packages may not be properly installed"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}   Installation Complete! ✓${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
print_info "To run the application:"
echo "   streamlit run app.py"
echo ""
print_info "Then open your browser to: http://localhost:8501"
echo ""
print_info "Additional commands:"
echo "   python3 scripts/evaluate_system.py  # Run system evaluation"
echo "   python3 scripts/retrain.py          # Retrain models"
echo "   streamlit run evaluation/user_survey.py  # User survey"
echo ""
echo "Press Enter to exit..."
read
