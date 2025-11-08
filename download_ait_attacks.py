#!/usr/bin/env python3
"""
AIT Dataset Attack Log Extractor

Downloads AIT Log Dataset files from Zenodo and extracts only attack-labeled logs
into a pandas DataFrame for easy analysis.

Usage:
    python download_ait_attacks.py --name=fox.zip
    python download_ait_attacks.py --name=harrison.zip --output=harrison_attacks.csv
    python download_ait_attacks.py --name=fox.zip --no-cleanup
"""

import argparse
import os
import sys
import json
import zipfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import requests
from tqdm import tqdm
import pandas as pd


class AITAttackExtractor:
    """Extracts attack logs from AIT Log Dataset"""
    
    BASE_URL = "https://zenodo.org/records/5789064/files/"
    DOWNLOAD_DIR = "ait_downloads"
    EXTRACT_DIR = "ait_extracted"
    
    def __init__(self, dataset_name: str, output_file: str = None, cleanup: bool = True):
        """
        Initialize the extractor
        
        Args:
            dataset_name: Name of the dataset file (e.g., 'fox.zip')
            output_file: Output CSV file path (default: <dataset>_attacks.csv)
            cleanup: Whether to clean up downloaded and extracted files after processing
        """
        self.dataset_name = dataset_name
        self.dataset_base = dataset_name.replace('.zip', '')
        self.output_file = output_file or f"{self.dataset_base}_attacks.csv"
        self.cleanup = cleanup
        
        # Create directories
        os.makedirs(self.DOWNLOAD_DIR, exist_ok=True)
        os.makedirs(self.EXTRACT_DIR, exist_ok=True)
        
        self.download_path = os.path.join(self.DOWNLOAD_DIR, dataset_name)
        self.extract_path = os.path.join(self.EXTRACT_DIR, self.dataset_base)
    
    def download_dataset(self) -> bool:
        """Download the dataset from Zenodo"""
        url = f"{self.BASE_URL}{self.dataset_name}?download=1"
        
        print(f"📥 Downloading {self.dataset_name} from Zenodo...")
        print(f"   URL: {url}")
        
        try:
            # Stream download with progress bar
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(self.download_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=self.dataset_name) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            print(f"✅ Download complete: {self.download_path}")
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Download failed: {e}")
            return False
    
    def extract_dataset(self) -> bool:
        """Extract the downloaded zip file"""
        print(f"\n📦 Extracting {self.dataset_name}...")
        
        try:
            with zipfile.ZipFile(self.download_path, 'r') as zip_ref:
                # Get list of files
                file_list = zip_ref.namelist()
                
                # Extract with progress bar
                with tqdm(total=len(file_list), desc="Extracting files") as pbar:
                    for file in file_list:
                        zip_ref.extract(file, self.extract_path)
                        pbar.update(1)
            
            print(f"✅ Extraction complete: {self.extract_path}")
            return True
            
        except zipfile.BadZipFile as e:
            print(f"❌ Extraction failed: {e}")
            return False
    
    def parse_labels(self, label_file_path: Path) -> Dict[int, Dict]:
        """
        Parse a label file and return a dictionary mapping line numbers to labels
        
        Args:
            label_file_path: Path to the label JSON file
            
        Returns:
            Dictionary: {line_number: {'labels': [...], 'rules': {...}}}
        """
        labels_dict = {}
        
        if not label_file_path.exists():
            return labels_dict
        
        try:
            with open(label_file_path, 'r') as f:
                for line in f:
                    if line.strip():
                        label_data = json.loads(line)
                        line_num = label_data['line']
                        labels_dict[line_num] = {
                            'labels': label_data.get('labels', []),
                            'rules': label_data.get('rules', {})
                        }
        except (json.JSONDecodeError, KeyError) as e:
            print(f"⚠️  Warning: Failed to parse {label_file_path}: {e}")
        
        return labels_dict
    
    def extract_attack_logs(self) -> pd.DataFrame:
        """
        Extract all attack-labeled logs from the dataset
        
        Returns:
            DataFrame with columns: log_file, host, log_type, line_number, timestamp, 
                                   log_content, attack_labels, attack_rules
        """
        print(f"\n🔍 Extracting attack logs from {self.dataset_base}...")
        
        # Find the dataset directory (it's inside the extracted folder)
        dataset_dir = Path(self.extract_path) / self.dataset_base
        if not dataset_dir.exists():
            # Sometimes it's directly in extract_path
            dataset_dir = Path(self.extract_path)
        
        gather_dir = dataset_dir / "gather"
        labels_dir = dataset_dir / "labels"
        
        if not gather_dir.exists():
            print(f"❌ Error: gather directory not found at {gather_dir}")
            return pd.DataFrame()
        
        if not labels_dir.exists():
            print(f"❌ Error: labels directory not found at {labels_dir}")
            return pd.DataFrame()
        
        attack_logs = []
        total_attack_lines = 0
        processed_files = 0
        
        # Walk through the labels directory to find all label files
        for label_file in labels_dir.rglob('*'):
            if label_file.is_file():
                # Get the relative path from labels directory
                rel_path = label_file.relative_to(labels_dir)
                
                # Find the corresponding log file in gather directory
                log_file = gather_dir / rel_path
                
                if not log_file.exists():
                    continue
                
                # Parse the labels for this file
                labels_dict = self.parse_labels(label_file)
                
                if not labels_dict:
                    continue
                
                processed_files += 1
                
                # Extract metadata from path
                path_parts = rel_path.parts
                host = path_parts[0] if len(path_parts) > 0 else "unknown"
                log_type = rel_path.stem
                
                # Read the log file and extract attack lines
                try:
                    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                        for line_num, log_line in enumerate(f, start=1):
                            if line_num in labels_dict:
                                total_attack_lines += 1
                                
                                # Extract timestamp if possible (first field usually)
                                timestamp = self._extract_timestamp(log_line)
                                
                                attack_logs.append({
                                    'log_file': str(rel_path),
                                    'host': host,
                                    'log_type': log_type,
                                    'line_number': line_num,
                                    'timestamp': timestamp,
                                    'log_content': log_line.strip(),
                                    'attack_labels': ','.join(labels_dict[line_num]['labels']),
                                    'attack_rules': json.dumps(labels_dict[line_num]['rules'])
                                })
                
                except Exception as e:
                    print(f"⚠️  Warning: Failed to read {log_file}: {e}")
        
        print(f"\n📊 Statistics:")
        print(f"   Files processed: {processed_files}")
        print(f"   Attack log lines extracted: {total_attack_lines}")
        
        # Create DataFrame
        df = pd.DataFrame(attack_logs)
        
        if not df.empty:
            print(f"\n✅ Created DataFrame with {len(df)} attack log entries")
            print(f"   Columns: {', '.join(df.columns)}")
            
            # Show label distribution
            if 'attack_labels' in df.columns:
                print(f"\n📈 Attack Label Distribution:")
                label_counts = df['attack_labels'].str.split(',').explode().value_counts()
                for label, count in label_counts.head(10).items():
                    print(f"   {label}: {count}")
        
        return df
    
    def _extract_timestamp(self, log_line: str) -> str:
        """
        Try to extract timestamp from log line
        
        Args:
            log_line: The log line content
            
        Returns:
            Extracted timestamp or empty string
        """
        # This is a simple heuristic - adjust based on log format
        parts = log_line.split()
        if len(parts) >= 2:
            # Try to return first two parts as timestamp (common in syslog)
            return ' '.join(parts[:2])
        return ""
    
    def save_dataframe(self, df: pd.DataFrame) -> None:
        """Save the DataFrame to CSV"""
        if df.empty:
            print("⚠️  No attack logs found. Nothing to save.")
            return
        
        print(f"\n💾 Saving to {self.output_file}...")
        df.to_csv(self.output_file, index=False)
        print(f"✅ Saved {len(df)} attack logs to {self.output_file}")
        
        # Also save a Parquet version for faster loading
        parquet_file = self.output_file.replace('.csv', '.parquet')
        df.to_parquet(parquet_file, index=False)
        print(f"✅ Also saved Parquet version: {parquet_file}")
    
    def cleanup_files(self) -> None:
        """Clean up downloaded and extracted files"""
        if not self.cleanup:
            print(f"\n📁 Files kept:")
            print(f"   Downloaded: {self.download_path}")
            print(f"   Extracted: {self.extract_path}")
            return
        
        print(f"\n🧹 Cleaning up temporary files...")
        
        # Remove downloaded zip
        if os.path.exists(self.download_path):
            os.remove(self.download_path)
            print(f"   ✓ Removed {self.download_path}")
        
        # Remove extracted directory
        if os.path.exists(self.extract_path):
            shutil.rmtree(self.extract_path)
            print(f"   ✓ Removed {self.extract_path}")
        
        print("✅ Cleanup complete")
    
    def run(self) -> pd.DataFrame:
        """Run the complete extraction pipeline"""
        print("=" * 80)
        print("🚀 AIT Dataset Attack Log Extractor")
        print("=" * 80)
        
        # Step 1: Download
        if not os.path.exists(self.download_path):
            if not self.download_dataset():
                return pd.DataFrame()
        else:
            print(f"✓ Using existing download: {self.download_path}")
        
        # Step 2: Extract
        if not os.path.exists(self.extract_path):
            if not self.extract_dataset():
                return pd.DataFrame()
        else:
            print(f"✓ Using existing extraction: {self.extract_path}")
        
        # Step 3: Extract attack logs
        df = self.extract_attack_logs()
        
        # Step 4: Save
        if not df.empty:
            self.save_dataframe(df)
        
        # Step 5: Cleanup
        self.cleanup_files()
        
        print("\n" + "=" * 80)
        print("✨ Done!")
        print("=" * 80)
        
        return df


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Download and extract attack logs from AIT Log Dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_ait_attacks.py --name=fox.zip
  python download_ait_attacks.py --name=harrison.zip --output=harrison_attacks.csv
  python download_ait_attacks.py --name=santos.zip --no-cleanup

Available datasets:
  fox.zip, harrison.zip, russellmitchell.zip, santos.zip, 
  shaw.zip, wardbeck.zip, wheeler.zip, wilson.zip
        """
    )
    
    parser.add_argument(
        '--name',
        type=str,
        required=True,
        help='Dataset name (e.g., fox.zip, harrison.zip)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output CSV file path (default: <dataset>_attacks.csv)'
    )
    
    parser.add_argument(
        '--no-cleanup',
        action='store_true',
        help='Keep downloaded and extracted files after processing'
    )
    
    args = parser.parse_args()
    
    # Validate dataset name
    valid_datasets = [
        'fox.zip', 'harrison.zip', 'russellmitchell.zip', 'santos.zip',
        'shaw.zip', 'wardbeck.zip', 'wheeler.zip', 'wilson.zip'
    ]
    
    if args.name not in valid_datasets:
        print(f"⚠️  Warning: '{args.name}' is not a known dataset name.")
        print(f"   Valid datasets: {', '.join(valid_datasets)}")
        print(f"   Proceeding anyway...")
    
    # Run extraction
    extractor = AITAttackExtractor(
        dataset_name=args.name,
        output_file=args.output,
        cleanup=not args.no_cleanup
    )
    
    df = extractor.run()
    
    # Exit code
    sys.exit(0 if not df.empty else 1)


if __name__ == "__main__":
    main()

