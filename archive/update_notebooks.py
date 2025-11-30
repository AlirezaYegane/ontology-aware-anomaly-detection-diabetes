"""
Script to update all notebooks with universal path setup cell
Modifies the first code cell in each notebook to use src.utils.setup_paths()
"""
import json
from pathlib import Path

# Universal path setup cell that will be injected
UNIVERSAL_PATH_CELL = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# === UNIVERSAL PATH SETUP (Works in both Local and Colab) ===\n",
        "import sys\n",
        "import os\n",
        "\n",
        "# Auto-detect environment and setup paths\n",
        "try:\n",
        "    from src.utils import setup_paths\n",
        "    env_type = setup_paths()\n",
        "except ImportError:\n",
        "    # Fallback if utils not found (first run)\n",
        "    print(\"⚙️  Setting up paths...\")\n",
        "    try:\n",
        "        import google.colab\n",
        "        in_colab = True\n",
        "        if 'notebooks' in os.getcwd():\n",
        "            os.chdir('..')\n",
        "        project_root = os.getcwd()\n",
        "        print(\"☁️  Detected: Google Colab\")\n",
        "    except ImportError:\n",
        "        in_colab = False\n",
        "        project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))\n",
        "        print(\"💻 Detected: Local Environment\")\n",
        "    \n",
        "    if project_root not in sys.path:\n",
        "        sys.path.insert(0, project_root)\n",
        "    print(f\"✅ Project root: {project_root}\")"
    ]
}

def update_notebook(notebook_path):
    """
    Update a notebook to include universal path setup as first code cell.
    If a path setup cell already exists, replace it.
    """
    print(f"Processing: {notebook_path}")
    
    # Load notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    cells = notebook['cells']
    
    # Find first code cell
    first_code_idx = None
    for i, cell in enumerate(cells):
        if cell['cell_type'] == 'code':
            first_code_idx = i
            break
    
    if first_code_idx is None:
        print(f"  ⚠️  No code cells found in {notebook_path}")
        return False
    
    # Check if first code cell is already a path setup cell
    first_code_content = ''.join(cells[first_code_idx]['source'])
    
    if 'sys.path' in first_code_content or 'COLAB' in first_code_content or 'PATH SETUP' in first_code_content:
        # Replace existing path setup cell
        cells[first_code_idx] = UNIVERSAL_PATH_CELL
        print(f"  ✅ Replaced existing path setup cell")
    else:
        # Insert new path setup cell before first code cell
        cells.insert(first_code_idx, UNIVERSAL_PATH_CELL)
        print(f"  ✅ Inserted new path setup cell")
    
    # Save notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=4)
    
    print(f"  💾 Saved {notebook_path}")
    return True

def main():
    """Update all notebooks in the notebooks/ directory"""
    print("="*80)
    print("UPDATING NOTEBOOKS WITH UNIVERSAL PATH SETUP")
    print("="*80)
    
    notebooks_dir = Path(__file__).parent / 'notebooks'
    
    if not notebooks_dir.exists():
        print(f"Error: notebooks/ directory not found at {notebooks_dir}")
        return 1
    
    # Find all .ipynb files
    notebooks = sorted(notebooks_dir.glob('*.ipynb'))
    
    if not notebooks:
        print("No notebooks found!")
        return 1
    
    print(f"\nFound {len(notebooks)} notebooks:\n")
    
    success_count = 0
    for notebook_path in notebooks:
        if update_notebook(notebook_path):
            success_count += 1
        print()
    
    print("="*80)
    print(f"COMPLETE: Updated {success_count}/{len(notebooks)} notebooks")
    print("="*80)
    
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())
