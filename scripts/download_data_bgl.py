"""
Download BGL Dataset

Downloads and extracts the BGL dataset from Zenodo.
"""

import os
import sys
import requests
import zipfile
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


def extract_zip(zip_path: str, extract_to: str) -> None:
    """
    Extract zip file.
    
    Args:
        zip_path: Path to zip file
        extract_to: Directory to extract to
    """
    print(f"\nExtracting: {zip_path}")
    print(f"Extract to: {extract_to}")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        members = zip_ref.infolist()
        with tqdm(desc="Extracting", total=len(members)) as pbar:
            for member in members:
                zip_ref.extract(member, path=extract_to)
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
    zip_filename = "BGL.zip"  # Fixed filename for BGL dataset
    zip_path = base_dir / zip_filename
    
    if zip_path.exists():
        print(f"\nFile already exists: {zip_path}")
        response = input("Download again? (y/n): ")
        if response.lower() != 'y':
            print("Skipping download")
        else:
            download_file(url, str(zip_path))
    else:
        download_file(url, str(zip_path))
    
    # Extract
    print()
    if not zip_path.exists():
        print(f"Error: Downloaded file not found at {zip_path}")
        sys.exit(1)
    
    extract_zip(str(zip_path), str(base_dir))
    
    # Verify files
    print("\n" + "="*60)
    print("Verifying extracted files")
    print("="*60)
    
    expected_files = [
        dataset_config['raw_log'],
        dataset_config['label_file']
    ]
    
    # Note: The actual files in BGL.zip might have different names
    # We'll check for common BGL file patterns
    possible_files = [
        'BGL.log',
        'BGL_anomaly_label.csv', 
        'anomaly_label.csv',
        'BGL.log_structured.csv'
    ]
    
    all_found = True
    found_files = []
    
    # Check for expected files
    for filename in expected_files:
        file_path = base_dir / filename
        if file_path.exists():
            size = file_path.stat().st_size
            print(f"✓ Found: {filename} ({size:,} bytes)")
            found_files.append(filename)
        else:
            print(f"✗ Missing: {filename}")
    
    # Check for alternative file names
    print("\nChecking for alternative BGL file names:")
    for filename in possible_files:
        file_path = base_dir / filename
        if file_path.exists() and filename not in found_files:
            size = file_path.stat().st_size
            print(f"✓ Found alternative: {filename} ({size:,} bytes)")
            found_files.append(filename)
    
    # Check if we have at least the raw log file
    log_files = [f for f in found_files if f.endswith('.log')]
    if not log_files:
        print("\n✗ No BGL log file found!")
        all_found = False
    else:
        print(f"\n✓ Found BGL log file: {log_files[0]}")
    
    # Check if we have label files
    label_files = [f for f in found_files if 'label' in f.lower() or 'anomaly' in f.lower()]
    if not label_files:
        print("⚠ Warning: No BGL label file found - may need manual labeling")
    else:
        print(f"✓ Found BGL label file: {label_files[0]}")
    
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
