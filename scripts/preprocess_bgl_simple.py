"""
BGL Simple Preprocessing

Works with the existing BGL structured files from LogHub.
"""

import os
import sys
from pathlib import Path
import yaml
import pandas as pd

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessing import TemplateMapper, DataSplitter, TextConverter


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def extract_block_id(content: str) -> str:
    """Extract block ID from BGL log content."""
    import re
    
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
    """Main function to run BGL preprocessing with existing structured files."""
    print("="*70)
    print("BGL SIMPLE PREPROCESSING (Using Existing Structured Files)")
    print("="*70)
    
    # Load configuration
    config = load_config()
    bgl_config = config['bgl_dataset']
    output_config = config['output']['bgl']
    preprocess_config = config['preprocessing']
    
    # Setup paths
    base_path = Path(bgl_config['base_path'])
    output_path = Path(output_config['base_path'])
    output_path.mkdir(parents=True, exist_ok=True)
    
    structured_csv = base_path / bgl_config['structured_file']
    templates_csv = base_path / bgl_config['templates_file']
    
    print(f"\nInput files:")
    print(f"  Structured: {structured_csv}")
    print(f"  Templates: {templates_csv}")
    print(f"\nOutput directory: {output_path}")
    
    # Verify files exist
    if not structured_csv.exists():
        print(f"Error: Structured file not found: {structured_csv}")
        sys.exit(1)
    
    if not templates_csv.exists():
        print(f"Error: Templates file not found: {templates_csv}")
        sys.exit(1)
    
    # Step 1: Load structured data
    print("\n" + "="*70)
    print("STEP 1: LOADING STRUCTURED DATA")
    print("="*70)
    
    df = pd.read_csv(structured_csv)
    print(f"Loaded {len(df):,} structured BGL entries")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check for required columns
    required_cols = ['Content', 'EventTemplate']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        sys.exit(1)
    
    # Step 2: Create template mapping
    print("\n" + "="*70)
    print("STEP 2: TEMPLATE MAPPING")
    print("="*70)
    
    # Load templates
    templates_df = pd.read_csv(templates_csv)
    print(f"Loaded {len(templates_df):,} templates")
    
    # Create mapping from templates
    unique_templates = df['EventTemplate'].unique()
    template_to_number = {template: idx + 1 for idx, template in enumerate(unique_templates)}
    
    print(f"Created mapping for {len(template_to_number):,} unique templates")
    
    # Add template numbers to dataframe
    df['TemplateNumber'] = df['EventTemplate'].map(template_to_number)
    
    # Step 3: Build sequences
    print("\n" + "="*70)
    print("STEP 3: BUILDING SEQUENCES")
    print("="*70)
    
    # Group by block ID
    block_sequences = {}
    logs_with_blocks = 0
    logs_without_blocks = 0
    
    for idx, row in df.iterrows():
        content = str(row['Content'])
        block_id = extract_block_id(content)
        
        if block_id:
            template_number = row['TemplateNumber']
            if block_id not in block_sequences:
                block_sequences[block_id] = []
            block_sequences[block_id].append(template_number)
            logs_with_blocks += 1
        else:
            logs_without_blocks += 1
        
        # Progress update
        if (idx + 1) % 100000 == 0:
            print(f"  Processed {idx + 1:,} logs, found {len(block_sequences):,} blocks")
    
    print(f"\nSequence building complete:")
    print(f"  Logs with block IDs: {logs_with_blocks:,}")
    print(f"  Logs without block IDs: {logs_without_blocks:,}")
    print(f"  Unique blocks: {len(block_sequences):,}")
    
    # Create sequences DataFrame
    sequences = []
    for block_id, template_numbers in block_sequences.items():
        sequences.append({
            'BlockId': block_id,
            'EventSequence': ' '.join(map(str, template_numbers)),
            'SequenceLength': len(template_numbers)
        })
    
    sequences_df = pd.DataFrame(sequences)
    
    # Step 4: Add labels
    print("\n" + "="*70)
    print("STEP 4: ADDING LABELS")
    print("="*70)
    
    # Check if Label column exists
    if 'Label' in df.columns:
        print("Found Label column in structured data")
        
        # Create block-based labels
        block_labels = {}
        
        for idx, row in df.iterrows():
            content = str(row['Content'])
            block_id = extract_block_id(content)
            
            if block_id:
                label = row['Label']
                
                # If block has any anomaly, mark entire block as anomalous
                if block_id not in block_labels:
                    block_labels[block_id] = label
                else:
                    # If current label is anomaly (1), keep it
                    if label == 1 or str(label).lower() in ['anomaly', 'true']:
                        block_labels[block_id] = 1
        
        # Convert to DataFrame
        block_label_df = pd.DataFrame([
            {'BlockId': block_id, 'Label': label}
            for block_id, label in block_labels.items()
        ])
        
        # Merge with sequences
        labeled_df = sequences_df.merge(block_label_df, on='BlockId', how='left')
        
        # Handle missing labels
        missing_labels = labeled_df['Label'].isna().sum()
        if missing_labels > 0:
            print(f"  Warning: {missing_labels} sequences without labels (will be dropped)")
            labeled_df = labeled_df.dropna(subset=['Label'])
        
        # Convert labels to int
        if labeled_df['Label'].dtype == 'object':
            label_mapping = {
                'Normal': 0, 'Anomaly': 1, 'normal': 0, 'anomaly': 1,
                0: 0, 1: 1, '0': 0, '1': 1,
                'False': 0, 'True': 1, False: 0, True: 1
            }
            labeled_df['Label'] = labeled_df['Label'].map(label_mapping)
        
        labeled_df['Label'] = labeled_df['Label'].astype(int)
        
    else:
        print("No Label column found - creating default labels (all normal)")
        labeled_df = sequences_df.copy()
        labeled_df['Label'] = 0
    
    # Statistics
    label_counts = labeled_df['Label'].value_counts().sort_index()
    print(f"\nLabel distribution:")
    for label, count in label_counts.items():
        label_name = "Normal" if label == 0 else "Anomaly"
        percentage = (count / len(labeled_df)) * 100
        print(f"  {label_name} ({label}): {count:,} ({percentage:.2f}%)")
    
    # Step 5: Convert to text
    print("\n" + "="*70)
    print("STEP 5: TEXT CONVERSION")
    print("="*70)
    
    # Create number to template mapping
    number_to_template = {v: k for k, v in template_to_number.items()}
    
    # Convert sequences to text
    def sequence_to_text(sequence_str):
        numbers = [int(x) for x in sequence_str.split()]
        templates = [number_to_template.get(num, f"<UNKNOWN_{num}>") for num in numbers]
        return preprocess_config['sequence_separator'].join(templates)
    
    labeled_df['Text'] = labeled_df['EventSequence'].apply(sequence_to_text)
    
    # Step 6: Train/Test Split
    print("\n" + "="*70)
    print("STEP 6: TRAIN/TEST SPLIT")
    print("="*70)
    
    splitter = DataSplitter(
        train_ratio=preprocess_config['train_ratio'],
        random_seed=preprocess_config['random_seed']
    )
    
    train_df, test_df = splitter.split_data(labeled_df)
    splitter.save_splits(train_df, test_df, str(output_path))
    
    # Final summary
    print("\n" + "="*70)
    print("BGL SIMPLE PREPROCESSING COMPLETE!")
    print("="*70)
    
    print("\nGenerated files:")
    files = [
        ("Labeled sequences", output_path / "bgl_sequence_labeled.csv"),
        ("Text sequences", output_path / "bgl_text.csv"),
        ("Train set", output_path / "bgl_train.csv"),
        ("Test set", output_path / "bgl_test.csv")
    ]
    
    for name, path in files:
        if path.exists():
            size = path.stat().st_size / (1024 * 1024)  # MB
            print(f"  ✓ {name}: {path.name} ({size:.2f} MB)")
        else:
            print(f"  ✗ {name}: {path.name} (not found)")
    
    print("\n" + "="*70)
    print("Ready for BGL benchmarking!")
    print("Run: python scripts/benchmark_bgl.py")
    print("="*70)


if __name__ == "__main__":
    main()
