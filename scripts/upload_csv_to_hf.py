"""
Upload already-saved CSV dataset to Hugging Face.
Use this if the main upload script was interrupted.
"""

import sys
from pathlib import Path
from datasets import Dataset

def main():
    """Upload CSV to Hugging Face."""
    print("="*70)
    print("UPLOAD CSV TO HUGGING FACE")
    print("="*70)
    
    # Check for CSV file
    csv_file = Path("datasets/ait/output/ait_raw_logs_dataset.csv")
    
    if not csv_file.exists():
        print(f"\n❌ CSV file not found: {csv_file}")
        print("Please run: python scripts/upload_raw_logs.py first")
        sys.exit(1)
    
    print(f"\n✅ Found CSV: {csv_file}")
    
    # Get repository name
    print("\n" + "="*70)
    print("HUGGING FACE UPLOAD SETUP")
    print("="*70)
    
    repo_name = input("\nEnter Hugging Face repository name (e.g., 'username/dataset-name'): ").strip()
    if not repo_name:
        print("❌ Repository name required.")
        sys.exit(1)
    
    print(f"\nRepository: {repo_name}")
    
    # Load dataset
    print("\n" + "="*70)
    print("LOADING DATASET")
    print("="*70)
    print(f"Loading from: {csv_file}")
    print("This may take a moment...")
    
    try:
        dataset = Dataset.from_csv(str(csv_file))
        print(f"✅ Dataset loaded: {len(dataset):,} entries")
        print(f"   Features: {dataset.features}")
    except Exception as e:
        print(f"\n❌ Failed to load dataset: {e}")
        sys.exit(1)
    
    # Upload
    print("\n" + "="*70)
    print("UPLOADING TO HUGGING FACE")
    print("="*70)
    
    try:
        print(f"Uploading to: {repo_name}")
        print("This may take a while...")
        
        dataset.push_to_hub(repo_name)
        
        print("\n✅ Upload complete!")
        print(f"✅ Dataset available at: https://huggingface.co/datasets/{repo_name}")
        
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        if "rate limit" in str(e).lower():
            print("\n⏰ Wait for rate limit to reset, then try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()

