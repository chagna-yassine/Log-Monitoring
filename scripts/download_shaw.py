"""
Download Shaw AIT dataset from Zenodo
"""

import os
import sys
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm
import yaml

def download_file(url: str, dest: Path) -> bool:
    """Download a file with progress bar."""
    print(f"Downloading: {url}")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(dest, 'wb') as f, tqdm(
            desc="Downloading",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        return True
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        return False

def extract_zip(zip_path: Path, extract_to: Path) -> bool:
    """Extract zip file."""
    print(f"Extracting {zip_path} to {extract_to}...")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            members = zip_ref.namelist()
            with tqdm(desc="Extracting", total=len(members)) as pbar:
                for member in members:
                    zip_ref.extract(member, path=extract_to)
                    pbar.update(1)
        
        print(f"[OK] Extraction complete: {extract_to}")
        return True
        
    except zipfile.BadZipFile as e:
        print(f"[ERROR] Extraction failed: {e}")
        return False

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    """Download Shaw AIT dataset."""
    print("="*70)
    print("DOWNLOAD SHAW AIT DATASET")
    print("="*70)
    
    # Load configuration
    config = load_config()
    ait_config = config['ait_dataset']
    
    dataset_name = "shaw"
    base_path = Path(ait_config['base_path'])
    
    # Create directory if it doesn't exist
    base_path.mkdir(parents=True, exist_ok=True)
    print(f"\n[OK] Created directory: {base_path}")
    
    # Setup paths
    archive_path = base_path / f"{dataset_name}.zip"
    extract_path = base_path
    
    # URL for Shaw dataset
    url = "https://zenodo.org/records/5789064/files/shaw.zip?download=1"
    
    print(f"\nDownloading: {url}")
    print(f"Target: {archive_path}")
    print()
    
    # Download
    if download_file(url, archive_path):
        print("\n[OK] Download complete")
        
        # Extract
        if extract_zip(archive_path, extract_path):
            print("\n[OK] Extraction complete")
            
            # Clean up
            if archive_path.exists():
                archive_path.unlink()
                print(f"[OK] Cleaned up {archive_path.name}")
            
            print(f"\n[OK] Dataset ready at: {base_path / dataset_name}")
            print(f"\nNow run:")
            print(f"  python scripts/upload_shaw_logs.py")
        else:
            print("\n[ERROR] Extraction failed")
    else:
        print("\n[ERROR] Download failed")

if __name__ == "__main__":
    main()

