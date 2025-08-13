# 🚀 Installation Guide - Hybrid Book Recommender

This guide provides multiple ways to install the Hybrid Book Recommender System on different operating systems.

## 📋 Prerequisites

- **Python 3.8 or higher**
- **Internet connection** (for downloading packages)
- **Administrator privileges** (recommended for Windows)

## 🖥️ Operating System Support

| OS | Script | Notes |
|---|---|---|
| **Windows** | `install.bat` | Auto-detects PowerShell, falls back to batch |
| **Windows** | `install_dependencies.ps1` | PowerShell script (recommended) |
| **Windows** | `install_dependencies.bat` | Basic batch script |
| **Linux/macOS** | `install_dependencies.sh` | Bash script |

## ⚡ Quick Installation

### **Windows Users (Recommended)**

1. **Double-click** `install.bat` in the project folder
2. **Or** right-click `install_dependencies.ps1` → "Run with PowerShell"

### **Linux/macOS Users**

1. **Open terminal** in the project folder
2. **Run**: `chmod +x install_dependencies.sh && ./install_dependencies.sh`

### **Manual Installation**

If scripts don't work, install manually:

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

## 🔧 Detailed Installation Steps

### **Windows Installation**

#### **Option 1: Auto-Install (Recommended)**
1. Navigate to the project folder
2. Double-click `install.bat`
3. Follow the prompts
4. Wait for installation to complete

#### **Option 2: PowerShell Script**
1. Right-click `install_dependencies.ps1`
2. Select "Run with PowerShell"
3. If prompted about execution policy, type `Y` and press Enter

#### **Option 3: Basic Batch Script**
1. Double-click `install_dependencies.bat`
2. Follow the console prompts

### **Linux Installation**

#### **Ubuntu/Debian**
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install python3 python3-pip build-essential python3-dev

# Run installer
chmod +x install_dependencies.sh
./install_dependencies.sh
```

#### **CentOS/RHEL/Fedora**
```bash
# Install system dependencies
sudo yum groupinstall 'Development Tools'
sudo yum install python3 python3-pip python3-devel

# Run installer
chmod +x install_dependencies.sh
./install_dependencies.sh
```

### **macOS Installation**

#### **Using Homebrew (Recommended)**
```bash
# Install Python
brew install python3

# Run installer
chmod +x install_dependencies.sh
./install_dependencies.sh
```

#### **Using Official Python**
1. Download Python from https://python.org
2. Install with "Add to PATH" option
3. Run the bash installer

## 🛠️ Troubleshooting

### **Common Issues**

#### **"Python is not installed"**
- **Windows**: Download from https://python.org, check "Add to PATH"
- **Linux**: `sudo apt-get install python3 python3-pip`
- **macOS**: `brew install python3`

#### **"pip is not available"**
- **Windows**: Reinstall Python with pip option
- **Linux**: `sudo apt-get install python3-pip`
- **macOS**: `brew install python3`

#### **Permission Errors**
- **Windows**: Run as Administrator
- **Linux/macOS**: Use `sudo` or fix permissions

#### **Build Errors**
- **Windows**: Install Visual Studio Build Tools
- **Linux**: `sudo apt-get install build-essential python3-dev`
- **macOS**: `xcode-select --install`

#### **Network Issues**
- Check internet connection
- Try using a different network
- Use VPN if behind corporate firewall

### **Advanced Troubleshooting**

#### **Virtual Environment (Recommended)**
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/macOS)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### **Manual Package Installation**
If specific packages fail:
```bash
# Install packages individually
pip install streamlit
pip install pandas
pip install numpy
pip install scikit-learn
pip install plotly
pip install joblib
```

#### **Alternative Package Managers**
```bash
# Using conda
conda install streamlit pandas numpy scikit-learn plotly joblib

# Using pipenv
pipenv install -r requirements.txt
```

## ✅ Verification

After installation, verify everything works:

```bash
# Test imports
python -c "import streamlit, pandas, numpy, sklearn, plotly, joblib; print('✓ All packages imported successfully')"

# Run the application
streamlit run app.py
```

## 🎯 Post-Installation

### **First Run**
1. Open terminal/command prompt
2. Navigate to project folder
3. Run: `streamlit run app.py`
4. Open browser to: `http://localhost:8501`

### **Additional Commands**
```bash
# System evaluation
python scripts/evaluate_system.py

# Model retraining
python scripts/retrain.py

# User survey
streamlit run evaluation/user_survey.py
```

## 📞 Support

If you encounter issues:

1. **Check the troubleshooting section above**
2. **Review error messages carefully**
3. **Try manual installation steps**
4. **Check system requirements**
5. **Update Python and pip to latest versions**

### **System Requirements**

- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space
- **Python**: 3.8 or higher
- **OS**: Windows 10+, macOS 10.14+, Ubuntu 18.04+

## 🔄 Updates

To update the system:

```bash
# Update pip
python -m pip install --upgrade pip

# Update packages
pip install --upgrade -r requirements.txt

# Or update individual packages
pip install --upgrade streamlit pandas numpy scikit-learn plotly joblib
```

---

**🎉 Installation Complete!** 

Your Hybrid Book Recommender System is now ready to use. Start exploring book recommendations at `http://localhost:8501`!
