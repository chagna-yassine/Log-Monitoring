"""
Complete Pipeline Runner

Runs the entire pipeline: download, preprocess, and benchmark.
"""

import sys
import subprocess
from pathlib import Path


def run_script(script_path: str, description: str) -> bool:
    """
    Run a Python script and handle errors.
    
    Args:
        script_path: Path to the script
        description: Description of the step
        
    Returns:
        True if successful, False otherwise
    """
    print("\n" + "="*70)
    print(f"RUNNING: {description}")
    print("="*70)
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=False
        )
        print(f"\n✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed with error code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n✗ {description} failed with error: {e}")
        return False


def main():
    """Run the complete pipeline."""
    print("="*70)
    print("LOG ANOMALY DETECTION BENCHMARKING")
    print("Complete Pipeline Runner")
    print("="*70)
    
    print("\nThis will run the following steps:")
    print("1. Download HDFS dataset from Zenodo")
    print("2. Preprocess logs (parse, create sequences, split)")
    print("3. Run benchmark with the model")
    
    response = input("\nContinue? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    scripts_dir = Path("scripts")
    
    # Step 1: Download data
    if not run_script(
        str(scripts_dir / "download_data.py"),
        "Step 1: Download Dataset"
    ):
        print("\nPipeline stopped due to error.")
        return
    
    # Step 2: Preprocess
    if not run_script(
        str(scripts_dir / "preprocess.py"),
        "Step 2: Preprocess Data"
    ):
        print("\nPipeline stopped due to error.")
        return
    
    # Step 3: Benchmark
    if not run_script(
        str(scripts_dir / "benchmark.py"),
        "Step 3: Run Benchmark"
    ):
        print("\nPipeline stopped due to error.")
        return
    
    # Success!
    print("\n" + "="*70)
    print("✓ COMPLETE PIPELINE FINISHED SUCCESSFULLY!")
    print("="*70)
    
    print("\nResults are available in the 'results/' directory.")
    print("Check 'benchmark_results.json' for detailed metrics.")


if __name__ == "__main__":
    main()

