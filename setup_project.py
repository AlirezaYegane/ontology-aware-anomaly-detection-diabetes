"""
Automated project setup script for the Ontology-Aware Anomaly Detection pipeline.

This script:
- Validates the Python version
- Creates a virtual environment (venv/)
- Upgrades pip inside the environment
- Installs dependencies from requirements.txt
- Prints clear instructions for next steps

It is safe to run multiple times; existing environments are reused.
"""

import sys
import platform
import subprocess
from pathlib import Path


# ---------------------------------------------------------------------------
# Simple console helpers
# ---------------------------------------------------------------------------

def print_header(text: str) -> None:
    """Print a formatted section header."""
    line = "=" * 80
    print(f"\n{line}")
    print(f"  {text}")
    print(line)


def print_success(text: str) -> None:
    """Print a success message."""
    print(f"[OK]  {text}")


def print_warning(text: str) -> None:
    """Print a warning message."""
    print(f"[WARN] {text}")


def print_error(text: str) -> None:
    """Print an error message."""
    print(f"[ERROR] {text}")


# ---------------------------------------------------------------------------
# Python version checks
# ---------------------------------------------------------------------------

def check_python_version() -> bool:
    """
    Check whether the current Python interpreter is in a recommended range.

    Recommended range: 3.8 <= version < 3.13
    (for good compatibility with PyTorch and the ML stack)
    """
    print_header("CHECKING PYTHON VERSION")

    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    print(f"Detected Python: {version_str}")

    if version.major < 3:
        print_error("Python 2.x is not supported.")
        print("Please install Python 3.8 or higher and rerun this script.")
        return False

    if version.minor < 8:
        print_warning(
            f"Python 3.{version.minor} is older than the recommended minimum (3.8)."
        )
        print("Some libraries may not be tested on this version.")
        response = input("Continue anyway? (y/n): ").strip().lower()
        if response != "y":
            return False

    if version.minor >= 13:
        print_warning(
            f"Python 3.{version.minor} is very new and may not be supported "
            "by PyTorch and other ML libraries."
        )
        print("Recommended range is Python 3.8 to 3.12.")
        response = input("Continue anyway? (y/n): ").strip().lower()
        if response != "y":
            return False

    print_success(f"Python {version_str} is acceptable for this project.")
    return True


# ---------------------------------------------------------------------------
# Virtual environment helpers
# ---------------------------------------------------------------------------

def create_virtual_environment() -> bool:
    """
    Create a local virtual environment in ./venv if it does not already exist.
    """
    print_header("VIRTUAL ENVIRONMENT SETUP")

    venv_path = Path("venv")

    if venv_path.exists():
        print_success("Virtual environment already exists (venv/).")
        return True

    print("Creating virtual environment in ./venv ...")
    try:
        subprocess.run(
            [sys.executable, "-m", "venv", "venv"],
            check=True,
        )
        print_success("Virtual environment created successfully.")
        return True
    except subprocess.CalledProcessError as exc:
        print_error(f"Failed to create virtual environment: {exc}")
        print("You may need to install the venv module for your Python distribution.")
        print("Examples:")
        print("  Ubuntu/Debian:  sudo apt-get install python3-venv")
        print("  macOS:          python3 -m pip install virtualenv")
        return False


def get_venv_python() -> Path:
    """Return the path to the Python executable inside the virtual environment."""
    if platform.system() == "Windows":
        return Path("venv") / "Scripts" / "python.exe"
    return Path("venv") / "bin" / "python"


def get_venv_pip() -> Path:
    """Return the path to the pip executable inside the virtual environment."""
    if platform.system() == "Windows":
        return Path("venv") / "Scripts" / "pip.exe"
    return Path("venv") / "bin" / "pip"


# ---------------------------------------------------------------------------
# Dependency management
# ---------------------------------------------------------------------------

def upgrade_pip() -> bool:
    """
    Upgrade pip inside the virtual environment to the latest version.
    """
    print_header("UPGRADING PIP")

    pip_exe = get_venv_pip()
    if not pip_exe.exists():
        print_warning("pip executable not found inside venv. Skipping pip upgrade.")
        return False

    print("Upgrading pip to the latest version inside venv ...")
    try:
        subprocess.run(
            [str(pip_exe), "install", "--upgrade", "pip"],
            check=True,
        )
        print_success("pip upgraded successfully.")
        return True
    except subprocess.CalledProcessError as exc:
        print_warning(f"Failed to upgrade pip (continuing anyway): {exc}")
        return False


def install_dependencies() -> bool:
    """
    Install project dependencies from requirements.txt inside the virtual environment.
    """
    print_header("INSTALLING DEPENDENCIES")

    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print_error("requirements.txt not found in the project root.")
        print("Make sure you run this script from the project root directory.")
        return False

    pip_exe = get_venv_pip()
    if not pip_exe.exists():
        print_error("pip executable inside venv not found.")
        return False

    print("Installing packages from requirements.txt...")
    print("This may take a few minutes, especially the first time.\n")

    try:
        subprocess.run(
            [str(pip_exe), "install", "-r", str(requirements_file)],
            check=True,
        )
        print_success("All dependencies installed successfully.")
        return True
    except subprocess.CalledProcessError as exc:
        print_error(f"Failed to install dependencies: {exc}")
        return False


# ---------------------------------------------------------------------------
# Final instructions
# ---------------------------------------------------------------------------

def print_activation_instructions() -> None:
    """
    Print instructions for activating the virtual environment
    and running the project.
    """
    print_header("SETUP COMPLETE")

    print("\nYour environment is ready to use.")
    print("\nNext steps:")

    print("\n1) Activate the virtual environment:")

    if platform.system() == "Windows":
        print("   Command Prompt:")
        print("       venv\\Scripts\\activate")
        print("   PowerShell:")
        print("       .\\venv\\Scripts\\Activate.ps1")
    else:
        print("   Bash / Zsh:")
        print("       source venv/bin/activate")

    print("\n2) Run the main pipeline script:")
    print("       python run_pipeline_direct.py")

    print("\n3) (Optional) Work with the notebooks:")
    print("       jupyter notebook")
    print("   Then open:")
    print("       notebooks/01_eda.ipynb")
    print("       notebooks/02_baseline_if.ipynb")
    print("       notebooks/04_ontology_eval.ipynb")

    print("\nAll generated results will be stored under the results/ directory.")
    print("\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    """Main setup routine."""
    print_header("ONTOLOGY-AWARE ANOMALY DETECTION - PROJECT SETUP")

    # 1. Check Python version
    if not check_python_version():
        print_error("Setup aborted due to Python version incompatibility.")
        return 1

    # 2. Create virtual environment
    if not create_virtual_environment():
        print_error("Setup aborted: could not create virtual environment.")
        return 1

    # 3. Upgrade pip (non-fatal if it fails)
    upgrade_pip()

    # 4. Install dependencies
    if not install_dependencies():
        print_error("Setup failed: dependency installation did not complete.")
        return 1

    # 5. Print final instructions
    print_activation_instructions()
    return 0


if __name__ == "__main__":
    sys.exit(main())
