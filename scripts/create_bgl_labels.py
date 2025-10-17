"""
Create BGL Labels

Creates anomaly labels for BGL sequences based on the structured log data.
This is useful when the BGL dataset doesn't have separate label files.
"""

import sys
import pandas as pd
from pathlib import Path
import yaml
import re


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_labels_from_logs(structured_csv: str, output_csv: str) -> None:
    """
    Create labels based on log content analysis.
    
    Args:
        structured_csv: Path to structured log CSV
        output_csv: Path to save labels CSV
    """
    print(f"Creating BGL labels from: {structured_csv}")
    
    # Load structured logs
    df = pd.read_csv(structured_csv)
    print(f"Loaded {len(df):,} structured log entries")
    
    # Extract block IDs and analyze content for anomalies
    block_labels = {}
    anomaly_keywords = [
        'error', 'fail', 'timeout', 'crash', 'exception',
        'abort', 'fatal', 'critical', 'panic', 'corrupt',
        'invalid', 'reject', 'deny', 'refuse', 'unavailable'
    ]
    
    print("Analyzing log content for anomaly indicators...")
    
    for idx, row in df.iterrows():
        content = str(row['Content']).lower()
        level = str(row['Level']).lower()
        
        # Extract block ID
        block_id = extract_block_id(content)
        if not block_id:
            continue
        
        # Determine if this log entry indicates an anomaly
        is_anomaly = False
        
        # Check log level
        if level in ['error', 'fatal', 'critical', 'panic']:
            is_anomaly = True
        
        # Check content for anomaly keywords
        if any(keyword in content for keyword in anomaly_keywords):
            is_anomaly = True
        
        # If block has any anomaly, mark entire block as anomalous
        if block_id not in block_labels:
            block_labels[block_id] = is_anomaly
        else:
            # If current entry is anomalous, mark block as anomalous
            if is_anomaly:
                block_labels[block_id] = True
        
        # Progress update
        if (idx + 1) % 100000 == 0:
            print(f"  Processed {idx + 1:,} logs, found {len(block_labels):,} blocks")
    
    # Create labels DataFrame
    labels_data = []
    for block_id, is_anomaly in block_labels.items():
        labels_data.append({
            'BlockId': block_id,
            'Label': 1 if is_anomaly else 0
        })
    
    labels_df = pd.DataFrame(labels_data)
    
    # Statistics
    anomaly_count = labels_df['Label'].sum()
    normal_count = len(labels_df) - anomaly_count
    anomaly_rate = (anomaly_count / len(labels_df)) * 100
    
    print(f"\nBGL label creation complete:")
    print(f"  Total blocks: {len(labels_df):,}")
    print(f"  Normal blocks: {normal_count:,} ({100-anomaly_rate:.2f}%)")
    print(f"  Anomalous blocks: {anomaly_count:,} ({anomaly_rate:.2f}%)")
    
    # Save labels
    print(f"\nSaving BGL labels to: {output_csv}")
    labels_df.to_csv(output_csv, index=False)
    
    return labels_df


def extract_block_id(content: str) -> str:
    """Extract block ID from log content."""
    # Try to extract various BGL identifiers
    patterns = [
        r'job\s+(\d+)',           # job 12345
        r'Job\s+(\d+)',           # Job 12345
        r'node\s+(\d+)',          # node 12345
        r'Node\s+(\d+)',          # Node 12345
        r'rank\s+(\d+)',          # rank 12345
        r'Rank\s+(\d+)',          # Rank 12345
        r'blk_(-?\d+)',           # block IDs (like HDFS)
        r'block\s+(\d+)',         # block 12345
    ]
    
    for pattern in patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            return f"bgl_{match.group(1)}"
    
    # If no specific pattern found, try to extract any numeric ID
    numeric_match = re.search(r'\b(\d{4,})\b', content)
    if numeric_match:
        return f"bgl_{numeric_match.group(1)}"
    
    return None


def main():
    """Main function to create BGL labels."""
    print("="*60)
    print("BGL LABEL CREATION")
    print("="*60)
    
    # Load configuration
    config = load_config()
    bgl_config = config['bgl_dataset']
    
    # Setup paths
    base_path = Path(bgl_config['base_path'])
    output_path = Path(config['output']['bgl']['base_path'])
    output_path.mkdir(parents=True, exist_ok=True)
    
    structured_csv = output_path / config['output']['bgl']['structured_log']
    labels_csv = base_path / bgl_config['label_file']
    
    # Check if structured log exists
    if not structured_csv.exists():
        print(f"Error: Structured log not found: {structured_csv}")
        print("Please run BGL preprocessing first.")
        sys.exit(1)
    
    # Create labels
    labels_df = create_labels_from_logs(str(structured_csv), str(labels_csv))
    
    print("\n" + "="*60)
    print("BGL LABEL CREATION COMPLETE!")
    print("="*60)
    
    print(f"\nCreated labels file: {labels_csv}")
    print("You can now run BGL preprocessing with these labels.")


if __name__ == "__main__":
    main()
