"""
Upload raw AIT structured logs to Hugging Face as a dataset.
This skips processing and uploads only the raw structured CSV files.
"""

import os
import sys
import shutil
from pathlib import Path
import pandas as pd
from huggingface_hub import HfApi, login
import yaml

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    """Upload raw structured logs to Hugging Face."""
    print("="*70)
    print("UPLOAD RAW AIT LOGS TO HUGGING FACE")
    print("="*70)
    
    # Load configuration
    config = load_config()
    ait_config = config['ait_dataset']
    
    # Get Hugging Face credentials
    print("\n" + "="*70)
    print("HUGGING FACE UPLOAD SETUP")
    print("="*70)
    
    print("\nTo upload raw logs to Hugging Face, you need:")
    print("1. A Hugging Face token (get from: https://huggingface.co/settings/tokens)")
    print("2. A repository name (e.g., 'username/ait-raw-logs')")
    print()
    
    # Get Hugging Face token
    hf_token = input("Enter your Hugging Face token: ").strip()
    if not hf_token:
        print("❌ No token provided. Upload cancelled.")
        sys.exit(1)
    
    print("✅ Token received!")
    
    # Get Hugging Face repository name
    repo_name = input("\nEnter Hugging Face repository name (e.g., 'username/ait-raw-logs'): ").strip()
    if not repo_name:
        print("❌ Repository name required. Upload cancelled.")
        sys.exit(1)
    
    print(f"\nRepository: {repo_name}")
    print(f"Token: ✅ Set")
    
    # Login to Hugging Face
    try:
        login(token=hf_token, add_to_git_credential=True)
        print("✅ Logged in to Hugging Face")
    except Exception as e:
        print(f"❌ Login failed: {e}")
        sys.exit(1)
    
    # Setup paths
    base_path = Path(ait_config['base_path'])
    dataset_name = ait_config['selected_dataset']
    
    # Check if dataset is in subdirectory or directly in base_path
    dataset_path = base_path / dataset_name
    if not dataset_path.exists():
        dataset_path = base_path
        print(f"\nDataset not found in subdirectory, using: {dataset_path}")
    
    gather_dir = dataset_path / 'gather'
    if not gather_dir.exists():
        print(f"\n❌ Error: gather directory not found: {gather_dir}")
        print("Please make sure the dataset is extracted.")
        sys.exit(1)
    
    # Initialize API
    api = HfApi()
    
    # Create repository if it doesn't exist
    try:
        api.create_repo(repo_id=repo_name, repo_type="dataset", exist_ok=True)
        print(f"\n✅ Repository ready: https://huggingface.co/datasets/{repo_name}")
    except Exception as e:
        print(f"⚠️  Repository creation note: {e}")
    
    # Process and upload raw log files
    print("\n" + "="*70)
    print("PROCESSING AND UPLOADING RAW LOGS")
    print("="*70)
    
    all_hosts = []
    total_files = 0
    
    # Get all hosts
    hosts = [d for d in gather_dir.iterdir() if d.is_dir()]
    print(f"\nFound {len(hosts)} hosts")
    
    # Process each host
    for host_dir in hosts:
        host_name = host_dir.name
        logs_dir = host_dir / 'logs'
        
        if not logs_dir.exists():
            continue
        
        print(f"\nProcessing host: {host_name}")
        
        # Get all log files
        log_files = []
        for log_type_dir in logs_dir.iterdir():
            if log_type_dir.is_dir():
                log_type = log_type_dir.name
                for log_file in log_type_dir.rglob('*'):
                    if log_file.is_file() and not log_file.name.endswith('.git'):
                        log_files.append((log_file, log_type))
        
        print(f"  Found {len(log_files)} log files")
        
        if log_files == 0:
            continue
        
        # Upload log files
        for log_file, log_type in log_files:
            total_files += 1
            file_size_mb = log_file.stat().st_size / (1024 * 1024)
            
            # Create upload path
            relative_path = log_file.relative_to(gather_dir)
            upload_path = f"raw_logs/{relative_path}"
            
            print(f"  [{total_files}] Uploading: {upload_path} ({file_size_mb:.2f} MB)")
            
            try:
                api.upload_file(
                    path_or_fileobj=str(log_file),
                    path_in_repo=upload_path,
                    repo_id=repo_name,
                    repo_type="dataset"
                )
                print(f"      ✅ Uploaded successfully")
            except Exception as e:
                print(f"      ❌ Upload failed: {e}")
                # Continue with next file
                continue
    
    # Upload dataset metadata
    print("\n" + "="*70)
    print("UPLOADING METADATA")
    print("="*70)
    
    # Try to upload dataset.yml if it exists
    dataset_yml = dataset_path / 'dataset.yml'
    if dataset_yml.exists():
        print("  Uploading dataset.yml...")
        try:
            api.upload_file(
                path_or_fileobj=str(dataset_yml),
                path_in_repo="dataset.yml",
                repo_id=repo_name,
                repo_type="dataset"
            )
            print("      ✅ dataset.yml uploaded")
        except Exception as e:
            print(f"      ⚠️  Could not upload dataset.yml: {e}")
    
    # Create a summary file
    summary = {
        "dataset": dataset_name,
        "total_files": total_files,
        "total_hosts": len(hosts),
        "hosts": [h.name for h in hosts]
    }
    
    summary_file = Path("upload_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Total files: {total_files}\n")
        f.write(f"Total hosts: {len(hosts)}\n")
        f.write(f"\nHosts:\n")
        for host in hosts:
            f.write(f"  - {host.name}\n")
    
    try:
        api.upload_file(
            path_or_fileobj=str(summary_file),
            path_in_repo="upload_summary.txt",
            repo_id=repo_name,
            repo_type="dataset"
        )
        print("  ✅ Upload summary created")
    except Exception as e:
        print(f"  ⚠️  Could not upload summary: {e}")
    finally:
        # Clean up
        if summary_file.exists():
            summary_file.unlink()
    
    # Final summary
    print("\n" + "="*70)
    print("UPLOAD COMPLETE!")
    print("="*70)
    print(f"✅ Repository: https://huggingface.co/datasets/{repo_name}")
    print(f"✅ Total files uploaded: {total_files}")
    print(f"✅ Total hosts: {len(hosts)}")
    print(f"\nDataset available at:")
    print(f"  https://huggingface.co/datasets/{repo_name}")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()

