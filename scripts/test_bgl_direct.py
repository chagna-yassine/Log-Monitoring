"""
Direct BGL Test

Tests BGL preprocessing directly with the manually downloaded files.
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


def extract_block_id(content: str) -> str:
    """Extract block ID from BGL log content."""
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
    """Test BGL preprocessing with manually downloaded files."""
    print("="*70)
    print("DIRECT BGL TEST")
    print("="*70)
    
    # Setup paths
    bgl_dir = Path("datasets/bgl")
    output_dir = Path("datasets/bgl/output/bgl")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    structured_file = bgl_dir / "BGL_2k.log_structured.csv"
    raw_log_file = bgl_dir / "BGL_2k.log"
    
    print(f"Input files:")
    print(f"  Structured: {structured_file}")
    print(f"  Raw log: {raw_log_file}")
    print(f"  Output: {output_dir}")
    
    # Check if files exist
    if not structured_file.exists():
        print(f"Error: Structured file not found: {structured_file}")
        return
    
    # Step 1: Load and examine structured data
    print("\n" + "="*70)
    print("STEP 1: LOADING STRUCTURED DATA")
    print("="*70)
    
    df = pd.read_csv(structured_file)
    print(f"Loaded {len(df):,} structured BGL entries")
    print(f"Columns: {df.columns.tolist()}")
    
    # Show sample data
    print(f"\nFirst 3 rows:")
    print(df.head(3))
    
    # Check for required columns
    if 'Content' not in df.columns:
        print("Error: 'Content' column not found")
        return
    
    if 'EventTemplate' not in df.columns:
        print("Warning: 'EventTemplate' column not found")
        # Create a simple template based on content
        print("Creating simple templates from content...")
        df['EventTemplate'] = df['Content'].apply(lambda x: str(x)[:50] + "...")
    
    # Step 2: Create template mapping
    print("\n" + "="*70)
    print("STEP 2: TEMPLATE MAPPING")
    print("="*70)
    
    unique_templates = df['EventTemplate'].unique()
    template_to_number = {template: idx + 1 for idx, template in enumerate(unique_templates)}
    
    print(f"Found {len(template_to_number):,} unique templates")
    
    # Add template numbers
    df['TemplateNumber'] = df['EventTemplate'].map(template_to_number)
    
    # Step 3: Extract block IDs and build sequences
    print("\n" + "="*70)
    print("STEP 3: BUILDING SEQUENCES")
    print("="*70)
    
    block_sequences = {}
    logs_with_blocks = 0
    logs_without_blocks = 0
    
    print("Extracting block IDs and building sequences...")
    
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
    
    # Statistics
    print(f"\nSequence statistics:")
    print(f"  Total sequences: {len(sequences_df):,}")
    print(f"  Average sequence length: {sequences_df['SequenceLength'].mean():.2f}")
    print(f"  Min sequence length: {sequences_df['SequenceLength'].min()}")
    print(f"  Max sequence length: {sequences_df['SequenceLength'].max()}")
    
    # Step 4: Handle labels
    print("\n" + "="*70)
    print("STEP 4: HANDLING LABELS")
    print("="*70)
    
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
    
    # Label statistics
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
        return ' | '.join(templates)  # Use | as separator
    
    labeled_df['Text'] = labeled_df['EventSequence'].apply(sequence_to_text)
    
    # Step 6: Train/Test Split
    print("\n" + "="*70)
    print("STEP 6: TRAIN/TEST SPLIT")
    print("="*70)
    
    # Simple 80/20 split
    train_ratio = 0.8
    train_size = int(len(labeled_df) * train_ratio)
    
    # Shuffle and split
    shuffled_df = labeled_df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_df = shuffled_df[:train_size]
    test_df = shuffled_df[train_size:]
    
    print(f"Train set size: {len(train_df):,} samples")
    print(f"Test set size: {len(test_df):,} samples")
    
    # Save files
    print("\n" + "="*70)
    print("STEP 7: SAVING FILES")
    print("="*70)
    
    # Save sequences
    sequences_csv = output_dir / "bgl_sequence.csv"
    sequences_df.to_csv(sequences_csv, index=False)
    print(f"✓ Saved sequences: {sequences_csv}")
    
    # Save labeled sequences
    labeled_csv = output_dir / "bgl_sequence_labeled.csv"
    labeled_df.to_csv(labeled_csv, index=False)
    print(f"✓ Saved labeled sequences: {labeled_csv}")
    
    # Save text sequences
    text_csv = output_dir / "bgl_text.csv"
    labeled_df[['BlockId', 'Text', 'Label']].to_csv(text_csv, index=False)
    print(f"✓ Saved text sequences: {text_csv}")
    
    # Save train/test splits
    train_csv = output_dir / "bgl_train.csv"
    test_csv = output_dir / "bgl_test.csv"
    
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    
    print(f"✓ Saved train set: {train_csv}")
    print(f"✓ Saved test set: {test_csv}")
    
    # Final summary
    print("\n" + "="*70)
    print("DIRECT BGL TEST COMPLETE!")
    print("="*70)
    
    print(f"\nGenerated files:")
    files = [
        ("Sequences", sequences_csv),
        ("Labeled sequences", labeled_csv),
        ("Text sequences", text_csv),
        ("Train set", train_csv),
        ("Test set", test_csv)
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
