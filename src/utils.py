"""
Utility functions for hybrid local/Colab environment support
Automatically detects environment and configures paths
"""
import sys
import os
from pathlib import Path

def setup_paths():
    """
    Universal path fixer for hybrid local/Colab execution.
    Automatically detects environment and adds project root to sys.path.
    
    Returns:
        str: Detected environment ('colab' or 'local')
    """
    # Check if running in Google Colab
    try:
        import google.colab
        in_colab = True
        env_type = 'colab'
    except ImportError:
        in_colab = False
        env_type = 'local'
    
    if in_colab:
        print("☁️  Detected: Google Colab Environment")
        
        # Try to find project root in Colab
        # Common patterns:
        # 1. /content/Ontology-aware Anomaly Detection Toy Pipeline/
        # 2. /content/ontology-anomaly-detection/
        # 3. Current directory if unzipped here
        
        possible_roots = [
            '/content/Ontology-aware Anomaly Detection Toy Pipeline',
            '/content',
        ]
        
        # Also check for any directory containing 'ontology' in current location
        try:
            current_dirs = [d for d in os.listdir('/content') 
                          if os.path.isdir(os.path.join('/content', d)) 
                          and 'ontology' in d.lower()]
            for dirname in current_dirs:
                possible_roots.insert(0, os.path.join('/content', dirname))
        except:
            pass
        
        # Find the first valid root (contains src/ directory)
        project_root = None
        for root in possible_roots:
            if os.path.exists(root) and os.path.exists(os.path.join(root, 'src')):
                project_root = root
                break
        
        if project_root is None:
            # Fallback: use current directory if it has src/
            if os.path.exists('src'):
                project_root = os.getcwd()
            else:
                # Last resort: try one level up
                parent = os.path.dirname(os.getcwd())
                if os.path.exists(os.path.join(parent, 'src')):
                    project_root = parent
                else:
                    project_root = '/content'
        
        # Change to project root
        os.chdir(project_root)
        
    else:
        print("💻 Detected: Local Environment")
        
        # For local: detect if we're in notebooks/ or root
        current_dir = os.getcwd()
        
        if 'notebooks' in current_dir or current_dir.endswith('notebooks'):
            # We're in notebooks directory, go up one level
            project_root = os.path.abspath(os.path.join(current_dir, '..'))
        else:
            # We're likely in root already
            project_root = current_dir
        
        # Verify src/ exists
        if not os.path.exists(os.path.join(project_root, 'src')):
            # Try parent directory
            parent = os.path.abspath(os.path.join(project_root, '..'))
            if os.path.exists(os.path.join(parent, 'src')):
                project_root = parent
    
    # Add project root to sys.path if not already there
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        print(f"✅ Added to sys.path: {project_root}")
    else:
        print(f"✅ Already in sys.path: {project_root}")
    
    # Verify we can import from src
    try:
        from src import preprocessing
        print("✅ Successfully imported from src/")
    except ImportError as e:
        print(f"⚠️  Warning: Could not import from src/: {e}")
        print(f"   Current sys.path: {sys.path[:3]}...")
    
    return env_type

def get_project_root():
    """
    Get the project root directory.
    
    Returns:
        Path: Project root directory
    """
    current = Path.cwd()
    
    # Check if current directory has src/
    if (current / 'src').exists():
        return current
    
    # Check if parent has src/
    if (current.parent / 'src').exists():
        return current.parent
    
    # Default to current
    return current

def ensure_results_dir():
    """
    Ensure results directory structure exists.
    Creates subdirectories for figures, models, and reports.
    """
    project_root = get_project_root()
    results_dir = project_root / 'results'
    
    # Create subdirectories
    (results_dir / 'figures').mkdir(parents=True, exist_ok=True)
    (results_dir / 'models').mkdir(parents=True, exist_ok=True)
    (results_dir / 'reports').mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Results directory ready: {results_dir}")
    
    return results_dir
