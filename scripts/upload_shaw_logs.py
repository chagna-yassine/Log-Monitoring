"""
Upload Shaw AIT logs to Hugging Face as a Dataset object.
This script processes the Shaw dataset from Zenodo.
"""

import os
import sys
import gc
from pathlib import Path
import pandas as pd
from datasets import Dataset
from huggingface_hub import login
import yaml

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    """Upload Shaw AIT logs to Hugging Face."""
    print("="*70)
    print("UPLOAD SHAW AIT LOGS TO HUGGING FACE")
    print("="*70)
    
    # Load configuration
    config = load_config()
    ait_config = config['ait_dataset']
    
    # Get Hugging Face credentials
    print("\n" + "="*70)
    print("HUGGING FACE UPLOAD SETUP")
    print("="*70)
    
    print("\nTo upload Shaw logs to Hugging Face, you need:")
    print("1. A Hugging Face token (get from: https://huggingface.co/settings/tokens)")
    print("2. A repository name (e.g., 'username/ait-shaw-raw-logs')")
    print()
    
    # Get Hugging Face token
    hf_token = input("Enter your Hugging Face token: ").strip()
    if not hf_token:
        print("❌ No token provided. Upload cancelled.")
        sys.exit(1)
    
    print("✅ Token received!")
    
    # Get Hugging Face repository name
    repo_name = input("\nEnter Hugging Face repository name (e.g., 'username/ait-shaw-raw-logs'): ").strip()
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
    
    # Setup paths - look directly in datasets/ait
    base_path = Path(ait_config['base_path'])
    
    # Check for dataset in the base path directly
    dataset_path = base_path
    gather_dir = dataset_path / 'gather'
    if not gather_dir.exists():
        print(f"\n❌ Error: gather directory not found: {gather_dir}")
        print("Please extract the Shaw dataset to: datasets/ait")
        print("Or make sure the dataset is extracted in: datasets/ait/gather")
        print("Download from: https://zenodo.org/records/5789064/files/shaw.zip")
        sys.exit(1)
    
    # Collect all log data
    print("\n" + "="*70)
    print("READING RAW LOG FILES")
    print("="*70)
    
    all_log_entries = []
    
    # Get all hosts
    hosts = [d for d in gather_dir.iterdir() if d.is_dir()]
    print(f"\nFound {len(hosts)} hosts")
    
    # Process each host
    total_files = 0
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
        
        # Read each log file
        for log_file, log_type in log_files:
            total_files += 1
            file_size_mb = log_file.stat().st_size / (1024 * 1024)
            
            # Create relative path identifier
            relative_path = log_file.relative_to(gather_dir)
            path_str = str(relative_path).replace('\\', '/')  # Normalize path separators
            
            # Identify binary files
            binary_extensions = ['.pcap', '.zip', '.xlsx', '.jpg', '.pdf', '.odt', '.docx']
            is_binary = any(log_file.name.endswith(ext) for ext in binary_extensions)
            file_ext = Path(log_file).suffix.lower()
            
            # For binary files, just record metadata
            if is_binary:
                print(f"  [{total_files}] Binary file (metadata only): {log_file.name} ({file_size_mb:.2f} MB)")
                all_log_entries.append({
                    'content': f"[BINARY FILE: {log_file.name}]",
                    'path': path_str,
                    'host': host_name,
                    'log_type': log_type,
                    'line_number': 0,
                    'file_size_mb': round(file_size_mb, 2),
                    'is_binary': True
                })
                continue
            
            # For very large JSON files, sample a subset
            if file_ext == '.json' and file_size_mb > 10:
                print(f"  [{total_files}] Large JSON (sampling): {log_file.name} ({file_size_mb:.2f} MB)")
                try:
                    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                        # Sample every Nth line for large files
                        sampling_rate = max(1, int(file_size_mb / 10))  # Sample more for larger files
                        for line_num, line in enumerate(f, 1):
                            if line_num % sampling_rate == 0:  # Sample every Nth line
                                if line.strip():
                                    all_log_entries.append({
                                        'content': line.strip(),
                                        'path': path_str,
                                        'host': host_name,
                                        'log_type': log_type,
                                        'line_number': line_num,
                                        'is_binary': False
                                    })
                    print(f"      ✅ Sampled lines from large JSON file")
                except Exception as e:
                    print(f"      ⚠️  Could not read file: {e}")
                continue
            
            # For text log files, read normally
            print(f"  [{total_files}] Reading: {log_file.name} ({file_size_mb:.2f} MB)")
            
            try:
                # For very large text files, use streaming with limits
                max_lines = 100000  # Limit to 100k lines per file to prevent memory issues
                
                if file_size_mb > 50:
                    print(f"      Using streaming mode for large file...")
                    lines_read = 0
                    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                        for line_num, line in enumerate(f, 1):
                            if line_num > max_lines:
                                break
                            if line.strip():
                                all_log_entries.append({
                                    'content': line.strip(),
                                    'path': path_str,
                                    'host': host_name,
                                    'log_type': log_type,
                                    'line_number': line_num,
                                    'is_binary': False
                                })
                                lines_read += 1
                    print(f"      ✅ Read {lines_read:,} lines from large file")
                else:
                    # For smaller files, read all lines
                    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                        for line_num, line in enumerate(f, 1):
                            if line.strip():
                                all_log_entries.append({
                                    'content': line.strip(),
                                    'path': path_str,
                                    'host': host_name,
                                    'log_type': log_type,
                                    'line_number': line_num,
                                    'is_binary': False
                                })
                    print(f"      ✅ Read log file")
                
                # Free memory periodically
                if total_files % 10 == 0:
                    gc.collect()
                    
            except Exception as e:
                print(f"      ⚠️  Could not read file: {e}")
                continue
    
    if not all_log_entries:
        print("\n❌ No log entries found!")
        sys.exit(1)
    
    # Convert to DataFrame for intermediate processing
    print("\n" + "="*70)
    print("PROCESSING DATA")
    print("="*70)
    print(f"Total log entries: {len(all_log_entries):,}")
    print(f"Converting to DataFrame...")
    
    # Free memory after creating list
    df = pd.DataFrame(all_log_entries)
    del all_log_entries  # Free memory
    gc.collect()
    
    print(f"✅ DataFrame created with {len(df):,} entries")
    print(f"   Columns: {list(df.columns)}")
    
    # Show sample of data
    print(f"\n📊 Sample entries:")
    print(df.head(3).to_string())
    
    # Save directly to CSV first
    output_path = Path("datasets/ait/output")
    output_path.mkdir(parents=True, exist_ok=True)
    
    csv_file = output_path / "shaw_raw_logs_dataset.csv"
    print(f"\n💾 Saving to CSV first: {csv_file}")
    print(f"   This may take a while for {len(df):,} entries...")
    
    # Save in chunks to avoid memory issues
    df.to_csv(csv_file, index=False, chunksize=100000)
    print(f"✅ Saved to CSV successfully!")
    
    # Free memory
    del df
    gc.collect()
    
    # Load from CSV to create Dataset (more memory efficient)
    print(f"\nLoading Dataset from CSV...")
    print(f"   (This is more memory efficient than creating from pandas)")
    
    dataset = Dataset.from_csv(str(csv_file))
    
    print(f"✅ Dataset created with {len(dataset):,} entries")
    print(f"   Features: {dataset.features}")
    
    # Upload to Hugging Face
    print("\n" + "="*70)
    print("UPLOADING TO HUGGING FACE")
    print("="*70)
    
    try:
        print(f"Uploading to: {repo_name}")
        print(f"Dataset size: {len(dataset):,} entries")
        print("This may take a while...")
        
        dataset.push_to_hub(repo_name)
        
        print("\n✅ Upload complete!")
        
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        print(f"\n💡 CSV file saved at: {csv_file}")
        print("You can upload manually later using:")
        print(f"   from datasets import Dataset")
        print(f"   dataset = Dataset.from_csv('{csv_file}')")
        print(f"   dataset.push_to_hub('{repo_name}')")
        if "rate limit" in str(e).lower():
            print("\n⏰ Wait for rate limit to reset, then try again.")
        sys.exit(1)
    
    # Final summary
    print("\n" + "="*70)
    print("UPLOAD COMPLETE!")
    print("="*70)
    print(f"✅ Repository: https://huggingface.co/datasets/{repo_name}")
    print(f"✅ Total entries: {len(dataset):,}")
    print(f"✅ Total files processed: {total_files}")
    print(f"✅ Total hosts: {len(hosts)}")
    print(f"\n📊 Dataset features:")
    print(f"   - content: The log line content (or '[BINARY FILE: filename]' for binary)")
    print(f"   - path: The source file path")
    print(f"   - host: The host name")
    print(f"   - log_type: The type of log")
    print(f"   - line_number: Line number in the file (0 for binary files)")
    print(f"   - is_binary: Whether the file is binary (True/False)")
    print(f"   - file_size_mb: File size in MB (for binary files)")
    print(f"\nDataset available at:")
    print(f"  https://huggingface.co/datasets/{repo_name}")
    print("\n" + "="*70)
    print("\n💡 To use the dataset later:")
    print(f"   from datasets import load_dataset")
    print(f"   dataset = load_dataset('{repo_name}')")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()

