#!/usr/bin/env python3
"""
AIT-LDS Chunked Processing Usage Example

This script demonstrates how to use the chunked preprocessing pipeline
to process large AIT datasets in 10GB chunks and upload to Hugging Face.
"""

import os
import sys
from pathlib import Path

# Add the scripts directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent))

from preprocess_ait_chunked import main as run_chunked_processing


def main():
    """Run the chunked preprocessing pipeline."""
    print("AIT-LDS Chunked Processing Pipeline")
    print("=" * 50)
    print()
    print("This pipeline will:")
    print("1. Process AIT dataset in 10GB chunks")
    print("2. Create sequences and extract templates")
    print("3. Upload each chunk to Hugging Face")
    print("4. Delete local files to save storage")
    print("5. Continue with next chunk")
    print()
    
    # Check if we're in the right directory
    if not Path("config.yaml").exists():
        print("Error: config.yaml not found!")
        print("Please run this script from the project root directory.")
        sys.exit(1)
    
    # Check if AIT dataset exists
    ait_path = Path("datasets/ait")
    if not ait_path.exists():
        print("Error: AIT dataset not found!")
        print("Please run: python scripts/download_data_ait.py")
        sys.exit(1)
    
    print("Prerequisites check:")
    print("✓ config.yaml found")
    print("✓ AIT dataset found")
    print()
    
    # Ask for confirmation
    response = input("Do you want to continue? (y/n): ").lower().strip()
    if response not in ['y', 'yes']:
        print("Processing cancelled.")
        sys.exit(0)
    
    # Run the chunked processing
    try:
        run_chunked_processing()
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
