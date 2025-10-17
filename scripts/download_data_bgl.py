"""
Download BGL Dataset

Downloads and extracts the BGL dataset from Zenodo.
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


def download_file(url: str, output_path: str) -> None:
    """
    Download a file with progress bar.
    
    Args:
        url: URL to download from
        output_path: Path to save the file
    """
    print(f"Downloading from: {url}")
    print(f"Saving to: {output_path}")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f, tqdm(
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
    
    print(f"Download complete: {output_path}")


def extract_tar_gz(tar_path: str, extract_to: str) -> None:
    """
    Extract tar.gz file.
    
    Args:
        tar_path: Path to tar.gz file
        extract_to: Directory to extract to
    """
    print(f"\nExtracting: {tar_path}")
    print(f"Extract to: {extract_to}")
    
    with tarfile.open(tar_path, 'r:gz') as tar:
        members = tar.getmembers()
        with tqdm(desc="Extracting", total=len(members)) as pbar:
            for member in members:
                tar.extract(member, path=extract_to)
                pbar.update(1)
    
    print("Extraction complete")


def main():
    """Main function to download and extract BGL dataset."""
    print("="*60)
    print("BGL Dataset Download")
    print("="*60)
    
    # Load configuration
    config = load_config()
    dataset_config = config['bgl_dataset']
    
    url = dataset_config['url']
    base_path = dataset_config['base_path']
    
    # Create base directory
    base_dir = Path(base_path)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Download file
    tar_filename = url.split('/')[-1]
    tar_path = base_dir / tar_filename
    
    if tar_path.exists():
        print(f"\nFile already exists: {tar_path}")
        response = input("Download again? (y/n): ")
        if response.lower() != 'y':
            print("Skipping download")
        else:
            download_file(url, str(tar_path))
    else:
        download_file(url, str(tar_path))
    
    # Extract
    print()
    if not tar_path.exists():
        print(f"Error: Downloaded file not found at {tar_path}")
        sys.exit(1)
    
    extract_tar_gz(str(tar_path), str(base_dir))
    
    # Verify files
    print("\n" + "="*60)
    print("Verifying extracted files")
    print("="*60)
    
    expected_files = [
        dataset_config['raw_log'],
        dataset_config['label_file']
    ]
    
    all_found = True
    for filename in expected_files:
        file_path = base_dir / filename
        if file_path.exists():
            size = file_path.stat().st_size
            print(f"✓ Found: {filename} ({size:,} bytes)")
        else:
            print(f"✗ Missing: {filename}")
            all_found = False
    
    if all_found:
        print("\n✓ All required BGL files found!")
        print("\nBGL dataset is ready for preprocessing.")
        print(f"Location: {base_dir.absolute()}")
    else:
        print("\n✗ Some files are missing!")
        print("Please check the extraction.")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(config['output']['bgl']['base_path'])
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nCreated BGL output directory: {output_dir.absolute()}")
    
    print("\n" + "="*60)
    print("BGL Download and setup complete!")
    print("="*60)


if __name__ == "__main__":
    main()
