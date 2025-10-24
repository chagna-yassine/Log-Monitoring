#!/usr/bin/env python3
"""
Example usage of the Hugging Face dataset uploader.
"""

import os
from pathlib import Path
from upload_to_huggingface import MemoryEfficientUploader

def upload_ait_fox_dataset():
    """Upload the AIT Fox dataset to Hugging Face Hub."""
    
    # Configuration
    DATASET_PATH = "datasets/ait/output/ait"
    REPO_NAME = "your-username/ait-fox-log-anomaly-dataset"  # Change this!
    
    # Check prerequisites
    print("Checking prerequisites...")
    
    # 1. Check if HF token is set
    if not os.getenv('HF_TOKEN'):
        print("❌ HF_TOKEN environment variable not set!")
        print("Please set your Hugging Face token:")
        print("  export HF_TOKEN=your_token_here")
        print("  or")
        print("  set HF_TOKEN=your_token_here  # Windows")
        return False
    
    # 2. Check if dataset directory exists
    dataset_path = Path(DATASET_PATH)
    if not dataset_path.exists():
        print(f"❌ Dataset directory not found: {dataset_path}")
        return False
    
    # 3. Check required files
    required_files = ["ait_train.csv", "ait_test.csv"]
    missing_files = []
    for file_name in required_files:
        if not (dataset_path / file_name).exists():
            missing_files.append(file_name)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    print("✅ Prerequisites check passed!")
    
    # Initialize uploader with memory-efficient settings
    uploader = MemoryEfficientUploader(
        dataset_path=DATASET_PATH,
        repo_name=REPO_NAME,
        chunk_size=3000,  # Conservative chunk size for 12GB RAM
        max_memory_gb=9.0  # Leave 3GB buffer for system
    )
    
    try:
        print(f"🚀 Starting upload to: {REPO_NAME}")
        print("This may take a while depending on dataset size...")
        
        # Upload dataset
        repo_url = uploader.upload_dataset(private=True)
        
        # Upload metadata files
        uploader.upload_metadata_files()
        
        print("="*70)
        print("🎉 UPLOAD COMPLETED SUCCESSFULLY!")
        print(f"📊 Dataset URL: {repo_url}")
        print("="*70)
        
        return True
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False

def main():
    """Main function."""
    print("Hugging Face Dataset Uploader")
    print("="*50)
    
    # Check system memory
    import psutil
    total_memory = psutil.virtual_memory().total / (1024**3)
    available_memory = psutil.virtual_memory().available / (1024**3)
    
    print(f"System Memory: {total_memory:.1f} GB total, {available_memory:.1f} GB available")
    
    if available_memory < 10:
        print("⚠️  Warning: Low available memory. Consider closing other applications.")
    
    # Upload dataset
    success = upload_ait_fox_dataset()
    
    if success:
        print("\n✅ Dataset uploaded successfully!")
        print("\nNext steps:")
        print("1. Visit your Hugging Face profile to see the dataset")
        print("2. Update the dataset description and tags")
        print("3. Share the dataset with your collaborators")
    else:
        print("\n❌ Upload failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
