"""
Preprocessing Pipeline

Runs the complete preprocessing pipeline for HDFS logs.
"""

import os
import sys
from pathlib import Path
import yaml

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessing import (
    HDFSLogParser,
    TemplateMapper,
    SequenceBuilder,
    DataSplitter,
    TextConverter
)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Main function to run preprocessing pipeline."""
    print("="*70)
    print("HDFS LOG PREPROCESSING PIPELINE")
    print("="*70)
    
    # Load configuration
    print("\nLoading configuration...")
    config = load_config()
    
    dataset_config = config['dataset']
    output_config = config['output']
    preprocess_config = config['preprocessing']
    
    # Setup paths
    base_path = Path(dataset_config['base_path'])
    output_path = Path(output_config['base_path'])
    output_path.mkdir(parents=True, exist_ok=True)
    
    raw_log = base_path / dataset_config['raw_log']
    label_file = base_path / dataset_config['label_file']
    
    structured_csv = output_path / output_config['structured_log']
    templates_csv = output_path / output_config['templates']
    sequences_csv = output_path / output_config['sequences']
    mapping_json = output_path / output_config['template_mapping']
    
    # Verify input files exist
    if not raw_log.exists():
        print(f"Error: Raw log file not found: {raw_log}")
        print("Please run download_data.py first.")
        sys.exit(1)
    
    if not label_file.exists():
        print(f"Error: Label file not found: {label_file}")
        print("Please run download_data.py first.")
        sys.exit(1)
    
    print(f"\nInput files:")
    print(f"  Raw log: {raw_log}")
    print(f"  Labels: {label_file}")
    print(f"\nOutput directory: {output_path}")
    
    # Step 1: Parse logs with Drain
    print("\n" + "="*70)
    print("STEP 1: LOG PARSING (Drain Algorithm)")
    print("="*70)
    
    drain_config = preprocess_config['drain']
    parser = HDFSLogParser(
        depth=drain_config['depth'],
        st=drain_config['st'],
        rex=drain_config['rex']
    )
    
    structured_df, templates_df = parser.parse_file(
        str(raw_log),
        str(structured_csv),
        str(templates_csv)
    )
    
    # Step 2: Create template mapping
    print("\n" + "="*70)
    print("STEP 2: TEMPLATE MAPPING")
    print("="*70)
    
    mapper = TemplateMapper()
    event_id_mapping = mapper.create_mapping(
        str(templates_csv),
        str(mapping_json)
    )
    
    # Step 3: Build block-wise sequences
    print("\n" + "="*70)
    print("STEP 3: BLOCK-WISE SEQUENCE CREATION")
    print("="*70)
    
    sequence_builder = SequenceBuilder()
    sequences_df = sequence_builder.build_sequences(
        str(structured_csv),
        event_id_mapping,
        str(sequences_csv)
    )
    
    # Step 4: Add labels
    print("\n" + "="*70)
    print("STEP 4: LABEL ASSIGNMENT")
    print("="*70)
    
    labeled_df = sequence_builder.add_labels(
        sequences_df,
        str(label_file)
    )
    
    # Save labeled sequences
    labeled_csv = output_path / "hdfs_sequence_labeled.csv"
    print(f"\nSaving labeled sequences to: {labeled_csv}")
    labeled_df.to_csv(labeled_csv, index=False)
    
    # Step 5: Convert to text
    print("\n" + "="*70)
    print("STEP 5: TEXT CONVERSION")
    print("="*70)
    
    # Load mapping for conversion
    mapper.load_mapping(str(mapping_json))
    
    text_converter = TextConverter(
        mapper,
        separator=preprocess_config['sequence_separator']
    )
    
    text_df = text_converter.convert_dataframe(labeled_df)
    
    # Save text version
    text_csv = output_path / "hdfs_text.csv"
    print(f"\nSaving text sequences to: {text_csv}")
    text_df.to_csv(text_csv, index=False)
    
    # Step 6: Train/Test Split
    print("\n" + "="*70)
    print("STEP 6: TRAIN/TEST SPLIT")
    print("="*70)
    
    splitter = DataSplitter(
        train_ratio=preprocess_config['train_ratio'],
        random_seed=preprocess_config['random_seed']
    )
    
    train_df, test_df = splitter.split_data(text_df)
    splitter.save_splits(train_df, test_df, str(output_path))
    
    # Final summary
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE!")
    print("="*70)
    
    print("\nGenerated files:")
    files = [
        ("Structured log", structured_csv),
        ("Event templates", templates_csv),
        ("Template mapping", mapping_json),
        ("Block sequences", sequences_csv),
        ("Labeled sequences", labeled_csv),
        ("Text sequences", text_csv),
        ("Train set", output_path / "hdfs_train.csv"),
        ("Test set", output_path / "hdfs_test.csv")
    ]
    
    for name, path in files:
        if path.exists():
            size = path.stat().st_size / (1024 * 1024)  # MB
            print(f"  ✓ {name}: {path.name} ({size:.2f} MB)")
        else:
            print(f"  ✗ {name}: {path.name} (not found)")
    
    print("\n" + "="*70)
    print("Ready for benchmarking!")
    print("Run: python scripts/benchmark.py")
    print("="*70)


if __name__ == "__main__":
    main()

