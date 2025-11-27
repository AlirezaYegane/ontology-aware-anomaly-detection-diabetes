"""
Run complete pipeline for Ontology-aware Anomaly Detection
Executes all notebooks in sequence and tracks results
"""
import os
import sys
import subprocess
from pathlib import Path

def run_notebook(notebook_path, output_path=None):
    """Execute a Jupyter notebook using subprocess"""
    if output_path is None:
        output_path = notebook_path.replace('.ipynb', '_executed.ipynb')
    
    print(f"\n{'='*80}")
    print(f"Executing: {notebook_path}")
    print(f"{'='*80}\n")
    
    cmd = [
        sys.executable, '-m', 'jupyter', 'nbconvert',
        '--to', 'notebook',
        '--execute',
        notebook_path,
        '--output', output_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ Successfully executed: {notebook_path}")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error executing {notebook_path}")
        print(f"Error: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    """Run all notebooks in sequence"""
    base_dir = Path(__file__).parent
    notebooks_dir = base_dir / 'notebooks'
    results_dir = base_dir / 'results'
    
    # Ensure results directory exists
    results_dir.mkdir(exist_ok=True)
    
    # Define notebook execution order
    notebooks = [
        '01_eda.ipynb',
        '02_baseline_if.ipynb',
        '03_autoencoder.ipynb',
        '04_ontology_eval.ipynb'
    ]
    
    print("\n" + "="*80)
    print("ONTOLOGY-AWARE ANOMALY DETECTION PIPELINE")
    print("="*80)
    print(f"Base directory: {base_dir}")
    print(f"Notebooks directory: {notebooks_dir}")
    print(f"Results directory: {results_dir}")
    print("="*80 + "\n")
    
    results = {}
    
    for notebook in notebooks:
        notebook_path = str(notebooks_dir / notebook)
        output_path = str(notebooks_dir / notebook.replace('.ipynb', '_executed.ipynb'))
        
        success = run_notebook(notebook_path, output_path)
        results[notebook] = success
        
        if not success:
            print(f"\n⚠️  Stopping pipeline due to error in {notebook}")
            break
    
    # Print summary
    print("\n" + "="*80)
    print("PIPELINE EXECUTION SUMMARY")
    print("="*80)
    for notebook, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{status}: {notebook}")
    print("="*80 + "\n")
    
    # Check results directory
    print("\nGenerated files in results/:")
    if results_dir.exists():
        for file in sorted(results_dir.iterdir()):
            print(f"  - {file.name} ({file.stat().st_size} bytes)")
    else:
        print("  (No results directory found)")
    
    # Return exit code
    all_success = all(results.values())
    return 0 if all_success else 1

if __name__ == '__main__':
    sys.exit(main())
