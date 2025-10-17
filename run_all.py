"""
Complete Pipeline Runner

Runs the entire pipeline: download, preprocess, and benchmark.
Now supports both HDFS and BGL datasets based on configuration.
"""

import sys
import subprocess
from pathlib import Path
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


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
    """Run the complete pipeline for the configured dataset."""
    print("="*70)
    print("LOG ANOMALY DETECTION BENCHMARKING")
    print("Complete Pipeline Runner")
    print("="*70)
    
    # Load configuration
    config = load_config()
    dataset_name = config.get('dataset_name', 'HDFS')
    
    print(f"\nDataset configured: {dataset_name}")
    
    if dataset_name == "HDFS":
        print("\nThis will run the HDFS pipeline:")
        print("1. Download HDFS dataset from Zenodo")
        print("2. Preprocess HDFS logs (parse, create sequences, split)")
        print("3. Run benchmark with the model")
        
        scripts_dir = Path("scripts")
        
        # Step 1: Download HDFS data
        if not run_script(
            str(scripts_dir / "download_data.py"),
            "Step 1: Download HDFS Dataset"
        ):
            print("\nPipeline stopped due to error.")
            return
        
        # Step 2: Preprocess HDFS
        if not run_script(
            str(scripts_dir / "preprocess.py"),
            "Step 2: Preprocess HDFS Data"
        ):
            print("\nPipeline stopped due to error.")
            return
        
        # Step 3: Benchmark HDFS
        if not run_script(
            str(scripts_dir / "benchmark.py"),
            "Step 3: Run HDFS Benchmark"
        ):
            print("\nPipeline stopped due to error.")
            return
            
        print("\nResults are available in the 'results/' directory.")
        print("Check 'benchmark_results.json' for detailed metrics.")
        
    elif dataset_name == "BGL":
        print("\nThis will run the BGL pipeline:")
        print("1. Download BGL dataset from Zenodo")
        print("2. Preprocess BGL logs (parse, create sequences, split)")
        print("3. Run benchmark with the model")
        
        scripts_dir = Path("scripts")
        
        # Step 1: Download BGL data
        if not run_script(
            str(scripts_dir / "download_data_bgl.py"),
            "Step 1: Download BGL Dataset"
        ):
            print("\nPipeline stopped due to error.")
            return
        
        # Step 2: Preprocess BGL
        if not run_script(
            str(scripts_dir / "preprocess_bgl_simple.py"),
            "Step 2: Preprocess BGL Data"
        ):
            print("\nPipeline stopped due to error.")
            return
        
        # Step 3: Benchmark BGL
        if not run_script(
            str(scripts_dir / "benchmark_bgl.py"),
            "Step 3: Run BGL Benchmark"
        ):
            print("\nPipeline stopped due to error.")
            return
            
        print("\nResults are available in the 'results/' directory.")
        print("Check 'bgl_benchmark_results.json' for detailed metrics.")
    
    else:
        print(f"\nError: Unknown dataset '{dataset_name}'")
        print("Please set dataset_name to 'HDFS' or 'BGL' in config.yaml")
        return
    
    # Success!
    print("\n" + "="*70)
    print(f"✓ COMPLETE {dataset_name} PIPELINE FINISHED SUCCESSFULLY!")
    print("="*70)


if __name__ == "__main__":
    main()

