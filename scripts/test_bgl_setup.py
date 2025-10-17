"""
Test BGL Setup

Quick test to verify BGL configuration and download works.
"""

import sys
from pathlib import Path
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Test BGL setup."""
    print("="*60)
    print("BGL SETUP TEST")
    print("="*60)
    
    # Load configuration
    config = load_config()
    dataset_name = config.get('dataset_name', 'HDFS')
    
    print(f"Dataset configured: {dataset_name}")
    
    if dataset_name != "BGL":
        print("⚠ Warning: Dataset is not set to BGL")
        print("Please set dataset_name: 'BGL' in config.yaml")
        return
    
    # Check BGL configuration
    bgl_config = config.get('bgl_dataset', {})
    print(f"\nBGL Configuration:")
    print(f"  Name: {bgl_config.get('name', 'Not set')}")
    print(f"  URL: {bgl_config.get('url', 'Not set')}")
    print(f"  Base Path: {bgl_config.get('base_path', 'Not set')}")
    print(f"  Raw Log: {bgl_config.get('raw_log', 'Not set')}")
    print(f"  Label File: {bgl_config.get('label_file', 'Not set')}")
    
    # Check if datasets directory exists
    datasets_dir = Path("datasets")
    if not datasets_dir.exists():
        print(f"\n✗ Datasets directory not found: {datasets_dir}")
        print("Please create it or run from project root")
        return
    
    # Check if BGL directory exists
    bgl_dir = Path(bgl_config.get('base_path', 'datasets/bgl'))
    if bgl_dir.exists():
        print(f"\n✓ BGL directory exists: {bgl_dir}")
        
        # List files in BGL directory
        files = list(bgl_dir.glob("*"))
        if files:
            print(f"  Files found:")
            for file in files:
                if file.is_file():
                    size = file.stat().st_size
                    print(f"    {file.name} ({size:,} bytes)")
                else:
                    print(f"    {file.name}/ (directory)")
        else:
            print(f"  Directory is empty")
    else:
        print(f"\n⚠ BGL directory does not exist: {bgl_dir}")
        print("This is normal if you haven't downloaded the dataset yet")
    
    print(f"\n" + "="*60)
    print("BGL SETUP TEST COMPLETE")
    print("="*60)
    
    print(f"\nNext steps:")
    print(f"1. Run: python scripts/download_data_bgl.py")
    print(f"2. Run: python scripts/preprocess_bgl.py")
    print(f"3. Run: python scripts/benchmark_bgl.py")
    print(f"Or run all at once: python run_all.py")


if __name__ == "__main__":
    main()
