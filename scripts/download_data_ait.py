"""
AIT-LDS Dataset Download

Downloads AIT Log Data Sets for intrusion detection evaluation.
Supports multiple testbeds: fox, harrison, russellmitchell, santos, shaw, wardbeck, wheeler, wilson
"""

import os
import sys
import requests
import tarfile
from pathlib import Path
from tqdm import tqdm
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def download_file(url: str, output_path: Path):
    """Downloads a file from a URL with a progress bar."""
    print(f"  Downloading {url} to {output_path}...")
    
    # Try different URL formats if the first one fails
    urls_to_try = [
        url,
        url.replace('/records/', '/record/'),
        url.replace('/record/', '/records/'),
        url.replace('?download=1', ''),
        url.replace('?download=1', '') + '?download=1'
    ]
    
    for i, try_url in enumerate(urls_to_try):
        if i > 0:
            print(f"  Trying alternative URL format: {try_url}")
        
        try:
            response = requests.get(try_url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192
            
            with open(output_path, 'wb') as f:
                with tqdm(total=total_size, unit='iB', unit_scale=True, desc=output_path.name) as pbar:
                    for chunk in response.iter_content(chunk_size=block_size):
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            print(f"  ✓ Download complete: {output_path}")
            return True
            
        except requests.exceptions.RequestException as e:
            if i == len(urls_to_try) - 1:  # Last attempt
                print(f"  ✗ All download attempts failed. Last error: {e}")
                return False
            else:
                print(f"  ⚠ Attempt {i+1} failed: {e}")
                continue
    
    return False


def extract_tar_gz(tar_path: Path, extract_to: Path) -> bool:
    """Extract tar.gz file."""
    print(f"  Extracting {tar_path} to {extract_to}...")
    
    try:
        with tarfile.open(tar_path, 'r:gz') as tar:
            members = tar.getmembers()
            with tqdm(desc="Extracting", total=len(members)) as pbar:
                for member in members:
                    tar.extract(member, path=extract_to)
                    pbar.update(1)
        
        print(f"  ✓ Extraction complete: {extract_to}")
        return True
        
    except tarfile.ReadError as e:
        print(f"  ✗ Extraction failed: {e}")
        return False


def verify_dataset_structure(dataset_path: Path, dataset_name: str) -> bool:
    """Verify that the extracted dataset has the expected structure."""
    print(f"  Verifying {dataset_name} dataset structure...")
    
    expected_dirs = ['gather', 'labels', 'processing', 'rules', 'environment']
    expected_files = ['dataset.yml']
    
    # Check for expected directories
    missing_dirs = []
    for dir_name in expected_dirs:
        dir_path = dataset_path / dir_name
        if not dir_path.exists():
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f"  ⚠ Missing directories: {missing_dirs}")
    
    # Check for dataset.yml
    dataset_yml = dataset_path / 'dataset.yml'
    if not dataset_yml.exists():
        print(f"  ⚠ Missing dataset.yml file")
        return False
    
    # Check gather directory structure
    gather_dir = dataset_path / 'gather'
    if gather_dir.exists():
        hosts = [d for d in gather_dir.iterdir() if d.is_dir()]
        print(f"  ✓ Found {len(hosts)} hosts in gather directory")
        
        # Check for logs in each host
        total_log_files = 0
        for host_dir in hosts:
            logs_dir = host_dir / 'logs'
            if logs_dir.exists():
                log_files = list(logs_dir.rglob('*'))
                log_files = [f for f in log_files if f.is_file()]
                total_log_files += len(log_files)
        
        print(f"  ✓ Found {total_log_files} log files across all hosts")
    
    # Check labels directory
    labels_dir = dataset_path / 'labels'
    if labels_dir.exists():
        label_files = list(labels_dir.rglob('*'))
        label_files = [f for f in label_files if f.is_file()]
        print(f"  ✓ Found {len(label_files)} label files")
    
    print(f"  ✓ Dataset structure verification complete")
    return True


def main():
    """Main function to download AIT-LDS dataset."""
    print("="*70)
    print("AIT-LDS DATASET DOWNLOAD")
    print("="*70)
    
    # Load configuration
    config = load_config()
    ait_config = config['ait_dataset']
    
    print(f"\nAIT-LDS Configuration:")
    print(f"  Name: {ait_config['name']}")
    print(f"  Base Path: {ait_config['base_path']}")
    print(f"  Selected Dataset: {ait_config['selected_dataset']}")
    
    # Setup paths
    base_path = Path(ait_config['base_path'])
    base_path.mkdir(parents=True, exist_ok=True)
    
    # Find selected dataset
    selected_dataset = None
    for dataset in ait_config['datasets']:
        if dataset['name'] == ait_config['selected_dataset']:
            selected_dataset = dataset
            break
    
    if not selected_dataset:
        print(f"Error: Selected dataset '{ait_config['selected_dataset']}' not found in configuration")
        print(f"Available datasets: {[d['name'] for d in ait_config['datasets']]}")
        sys.exit(1)
    
    print(f"\nSelected Dataset: {selected_dataset['name']}")
    print(f"  URL: {selected_dataset['url']}")
    print(f"  Size: {selected_dataset['size']}")
    print(f"  Scan Volume: {selected_dataset['scan_volume']}")
    
    # Check if dataset already exists
    dataset_dir = base_path / selected_dataset['name']
    if dataset_dir.exists() and any(dataset_dir.iterdir()):
        print(f"\nDataset directory already exists: {dataset_dir}")
        response = input("Download again? (y/n): ")
        if response.lower() != 'y':
            print("Skipping download")
            if verify_dataset_structure(dataset_dir, selected_dataset['name']):
                print(f"\n✓ Dataset is ready for preprocessing!")
                return
            else:
                print("Dataset structure verification failed. Re-downloading...")
    
    # Download dataset
    print(f"\n" + "="*70)
    print("DOWNLOADING DATASET")
    print("="*70)
    
    archive_name = f"{selected_dataset['name']}.tar.gz"
    archive_path = base_path / archive_name
    
    if archive_path.exists():
        print(f"Archive already exists: {archive_path}")
        response = input("Re-download? (y/n): ")
        if response.lower() != 'y':
            print("Using existing archive")
        else:
            archive_path.unlink()
    
    if not archive_path.exists():
        success = download_file(selected_dataset['url'], archive_path)
        if not success:
            print("Download failed. Exiting.")
            sys.exit(1)
    
    # Extract dataset
    print(f"\n" + "="*70)
    print("EXTRACTING DATASET")
    print("="*70)
    
    success = extract_tar_gz(archive_path, base_path)
    if not success:
        print("Extraction failed. Exiting.")
        sys.exit(1)
    
    # Verify extraction
    print(f"\n" + "="*70)
    print("VERIFYING DATASET")
    print("="*70)
    
    if verify_dataset_structure(dataset_dir, selected_dataset['name']):
        print(f"\n✓ Dataset downloaded and verified successfully!")
        
        # Clean up archive if extraction was successful
        if archive_path.exists():
            print(f"Cleaning up archive: {archive_path}")
            archive_path.unlink()
        
        print(f"\nDataset ready for preprocessing:")
        print(f"  Location: {dataset_dir.absolute()}")
        print(f"  Next step: python scripts/preprocess_ait.py")
        
    else:
        print(f"\n✗ Dataset verification failed!")
        print("Please check the download and extraction.")
        sys.exit(1)
    
    # Show dataset info
    print(f"\n" + "="*70)
    print("DATASET INFORMATION")
    print("="*70)
    
    print(f"\nDataset: {selected_dataset['name']}")
    print(f"Simulation time: Check dataset.yml file")
    print(f"Attack time: Check dataset.yml file")
    print(f"Scan volume: {selected_dataset['scan_volume']}")
    print(f"Unpacked size: {selected_dataset['size']}")
    
    print(f"\nAvailable datasets for future use:")
    for dataset in ait_config['datasets']:
        status = "✓ Downloaded" if dataset['name'] == selected_dataset['name'] else "Available"
        print(f"  {dataset['name']}: {dataset['size']} ({status})")
    
    print(f"\n" + "="*70)
    print("AIT-LDS DOWNLOAD COMPLETE!")
    print("="*70)
    
    print(f"\nNote: AIT-LDS datasets are large (14-39 GB each).")
    print(f"If download fails, you can:")
    print(f"  1. Check your internet connection")
    print(f"  2. Try downloading manually from Zenodo")
    print(f"  3. Use a different dataset (edit config.yaml)")
    print(f"  4. Check Zenodo record: https://zenodo.org/record/5789063")


if __name__ == "__main__":
    main()
