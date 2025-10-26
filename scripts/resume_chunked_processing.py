#!/usr/bin/env python3
"""
Resume AIT-LDS Chunked Processing

This script allows you to resume the chunked processing from where it left off.
It will skip already processed chunks and continue with the remaining data.
"""

import os
import sys
from pathlib import Path

# Add the scripts directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent))

from preprocess_ait_chunked import main as run_chunked_processing


def main():
    """Resume the chunked preprocessing pipeline."""
    print("AIT-LDS Chunked Processing - RESUME MODE")
    print("=" * 50)
    print()
    print("This will resume processing from where it left off.")
    print("Already processed chunks will be skipped.")
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
    response = input("Do you want to resume processing? (y/n): ").lower().strip()
    if response not in ['y', 'yes']:
        print("Processing cancelled.")
        sys.exit(0)
    
    # Run the chunked processing
    try:
        run_chunked_processing()
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
        print("You can resume later by running this script again.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during processing: {e}")
        print("You can resume later by running this script again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
