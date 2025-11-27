"""
Automated Project Setup Script for Ontology-aware Anomaly Detection
Handles virtual environment creation, dependency installation, and Python version validation
"""
import sys
import os
import subprocess
import platform
from pathlib import Path

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80)

def print_success(text):
    """Print success message"""
    print(f"✅ {text}")

def print_warning(text):
    """Print warning message"""
    print(f"⚠️  {text}")

def print_error(text):
    """Print error message"""
    print(f"❌ {text}")

def check_python_version():
    """Check if Python version is compatible"""
    print_header("CHECKING PYTHON VERSION")
    
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    print(f"Detected Python: {version_str}")
    
    # Warn if version is outside recommended range
    if version.major < 3:
        print_error("Python 2.x is not supported!")
        print("Please upgrade to Python 3.8 or higher.")
        return False
    
    if version.minor < 8:
        print_warning(f"Python 3.{version.minor} is older than recommended.")
        print("Please consider upgrading to Python 3.8 or higher.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return False
    
    if version.minor >= 13:
        print_warning(f"Python 3.{version.minor} is very new and may have compatibility issues.")
        print("PyTorch and other ML libraries may not support Python 3.13+")
        print("Recommended: Use Python 3.8 - 3.12")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return False
    
    print_success(f"Python {version_str} is compatible")
    return True

def create_virtual_environment():
    """Create virtual environment if it doesn't exist"""
    print_header("VIRTUAL ENVIRONMENT SETUP")
    
    venv_path = Path("venv")
    
    if venv_path.exists():
        print_success("Virtual environment already exists")
        return True
    
    print("Creating virtual environment...")
    try:
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print_success("Virtual environment created successfully")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to create virtual environment: {e}")
        print("\nTry installing python3-venv:")
        print("  Ubuntu/Debian: sudo apt-get install python3-venv")
        print("  macOS: python3 -m pip install virtualenv")
        return False

def get_venv_python():
    """Get path to Python executable in virtual environment"""
    if platform.system() == "Windows":
        return Path("venv") / "Scripts" / "python.exe"
    else:
        return Path("venv") / "bin" / "python"

def get_pip_executable():
    """Get path to pip executable in virtual environment"""
    if platform.system() == "Windows":
        return Path("venv") / "Scripts" / "pip.exe"
    else:
        return Path("venv") / "bin" / "pip"

def upgrade_pip():
    """Upgrade pip to latest version"""
    print_header("UPGRADING PIP")
    
    pip_exe = get_pip_executable()
    
    print("Upgrading pip to latest version...")
    try:
        subprocess.run([str(pip_exe), "install", "--upgrade", "pip"], 
                      check=True, capture_output=True)
        print_success("Pip upgraded successfully")
        return True
    except subprocess.CalledProcessError as e:
        print_warning("Failed to upgrade pip, continuing anyway...")
        return True

def install_dependencies():
    """Install project dependencies from requirements.txt"""
    print_header("INSTALLING DEPENDENCIES")
    
    requirements_file = Path("requirements.txt")
    
    if not requirements_file.exists():
        print_error("requirements.txt not found!")
        return False
    
    pip_exe = get_pip_executable()
    
    print("Installing packages from requirements.txt...")
    print("This may take several minutes...\n")
    
    try:
        # Install with progress output
        subprocess.run(
            [str(pip_exe), "install", "-r", "requirements.txt"],
            check=True
        )
        print_success("All dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install dependencies: {e}")
        return False

def print_activation_instructions():
    """Print instructions for activating virtual environment"""
    print_header("SETUP COMPLETE!")
    
    print("\n🎉 Your project is ready to use!")
    print("\n📋 Next steps:")
    print("\n1️⃣  Activate the virtual environment:")
    
    if platform.system() == "Windows":
        print("   Command: .\\venv\\Scripts\\activate")
        print("   PowerShell: .\\venv\\Scripts\\Activate.ps1")
    else:
        print("   Command: source venv/bin/activate")
    
    print("\n2️⃣  Start Jupyter Notebook:")
    print("   Command: jupyter notebook")
    
    print("\n3️⃣  Open and run notebooks in order:")
    print("   - notebooks/01_eda.ipynb")
    print("   - notebooks/02_baseline_if.ipynb")
    print("   - notebooks/03_autoencoder.ipynb")
    print("   - notebooks/04_ontology_eval.ipynb")
    
    print("\n" + "="*80 + "\n")

def main():
    """Main setup routine"""
    print("\n" + "="*80)
    print("  ONTOLOGY-AWARE ANOMALY DETECTION - PROJECT SETUP")
    print("="*80)
    
    # Step 1: Check Python version
    if not check_python_version():
        print_error("Setup aborted due to Python version incompatibility")
        return 1
    
    # Step 2: Create virtual environment
    if not create_virtual_environment():
        print_error("Setup aborted - could not create virtual environment")
        return 1
    
    # Step 3: Upgrade pip
    if not upgrade_pip():
        print_warning("Continuing without pip upgrade...")
    
    # Step 4: Install dependencies
    if not install_dependencies():
        print_error("Setup failed - could not install dependencies")
        return 1
    
    # Step 5: Print activation instructions
    print_activation_instructions()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
