"""
AIT-LDS Preprocessing Pipeline

Complete preprocessing pipeline for AIT Log Data Sets.
Handles multi-host logs, attack labels, and sequence building.
"""

import os
import sys
from pathlib import Path
import yaml
import pandas as pd

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

from parsers.drain import LogParser as DrainParser
from preprocessing.ait_parser import AITLogParser
from preprocessing.template_mapper import TemplateMapper
from preprocessing.ait_sequence_builder import AITSequenceBuilder
from preprocessing.data_splitter import DataSplitter
from preprocessing.text_converter import TextConverter


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Main function to run AIT-LDS preprocessing pipeline."""
    print("="*70)
    print("AIT-LDS LOG ANOMALY DETECTION PREPROCESSING")
    print("="*70)

    # Load configuration
    config = load_config()
    ait_config = config['ait_dataset']
    output_config = config['output']['ait']
    drain_config = config['preprocessing']['drain']
    preprocess_config = config['preprocessing']
    
    # Setup paths
    base_path = Path(ait_config['base_path'])
    dataset_name = ait_config['selected_dataset']
    dataset_path = base_path / dataset_name
    output_path = Path(output_config['base_path'])
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nAIT-LDS Configuration:")
    print(f"  Dataset: {dataset_name}")
    print(f"  Dataset Path: {dataset_path}")
    print(f"  Output Path: {output_path}")

    # Verify dataset exists
    if not dataset_path.exists():
        print(f"Error: Dataset not found: {dataset_path}")
        print("Please run: python scripts/download_data_ait.py")
        sys.exit(1)

    # Step 1: AIT Log Parsing (multi-host structure)
    print("\n" + "="*70)
    print("STEP 1: AIT-LDS LOG PARSING")
    print("="*70)
    
    ait_parser = AITLogParser(config)
    structured_df = ait_parser.parse_all_logs()
    
    if structured_df.empty:
        print("Error: No logs were parsed from AIT-LDS dataset")
        sys.exit(1)
    
    print(f"Parsed {len(structured_df):,} AIT-LDS log entries")
    
    # Save structured logs
    structured_csv = output_path / output_config['structured_log']
    ait_parser.save_structured_logs(structured_df)

    # Step 2: Event Template Extraction (using Drain)
    print("\n" + "="*70)
    print("STEP 2: EVENT TEMPLATE EXTRACTION (DRAIN)")
    print("="*70)
    
    # Prepare content for Drain
    content_file = output_path / "ait_content_for_drain.log"
    structured_df['Content'].to_csv(content_file, index=False, header=False)
    
    drain_parser = DrainParser(
        log_format=drain_config['log_format'],
        indir=str(output_path),
        outdir=str(output_path),
        rex=drain_config['regex'],
        keep_para=drain_config['keep_para'],
        depth=drain_config['depth'],
        st=drain_config['st']
    )
    
    print(f"Extracting templates from {content_file} using Drain...")
    drain_parser.parse(
        logName=content_file.name,
        savePath=str(output_path / output_config['structured_log'])
    )
    
    # Load Drain-processed structured log
    drain_structured_df = pd.read_csv(output_path / output_config['structured_log'])
    
    # Merge EventId and EventTemplate back into our structured_df
    if len(structured_df) != len(drain_structured_df):
        print("  Warning: Mismatch in row count after Drain parsing")
        print(f"  Original: {len(structured_df)}, Drain: {len(drain_structured_df)}")
        
        # Try to align by content
        structured_df = structured_df.reset_index(drop=True)
        drain_structured_df = drain_structured_df.reset_index(drop=True)
        
        # Assume order is preserved, pad or truncate as needed
        min_len = min(len(structured_df), len(drain_structured_df))
        structured_df = structured_df.iloc[:min_len]
        drain_structured_df = drain_structured_df.iloc[:min_len]
    
    structured_df['EventId'] = drain_structured_df['EventId']
    structured_df['EventTemplate'] = drain_structured_df['EventTemplate']
    
    # Clean up temporary file
    content_file.unlink(missing_ok=True)
    
    print(f"Extracted {structured_df['EventTemplate'].nunique():,} unique event templates")
    
    # Save updated structured logs
    structured_df.to_csv(structured_csv, index=False)

    # Step 3: Template Mapping
    print("\n" + "="*70)
    print("STEP 3: TEMPLATE MAPPING")
    print("="*70)
    
    template_mapper = TemplateMapper(config, dataset_type='ait')
    template_mapper.create_mapping(structured_df)
    print(f"Template mapping saved")

    # Step 4: Sequence Creation & Label Assignment
    print("\n" + "="*70)
    print("STEP 4: SEQUENCE CREATION & LABEL ASSIGNMENT")
    print("="*70)
    
    ait_sequence_builder = AITSequenceBuilder(config)
    
    # Load template mapping
    template_mapping_file = output_path / output_config['template_mapping']
    with open(template_mapping_file, 'r') as f:
        template_mapping = json.load(f)
    
    # Create event ID to template number mapping
    event_id_mapping = {}
    for template_id, template_content in template_mapping.items():
        # Find EventId for this template
        matching_rows = structured_df[structured_df['EventTemplate'] == template_content]
        if not matching_rows.empty:
            event_id = matching_rows['EventId'].iloc[0]
            event_id_mapping[int(event_id)] = int(template_id)
    
    sequences_csv = output_path / output_config['sequences']
    sequences_df = ait_sequence_builder.build_sequences(
        str(structured_csv),
        event_id_mapping,
        str(sequences_csv)
    )
    
    # Add labels to sequences
    labeled_sequences_df = ait_sequence_builder.add_labels(sequences_df, str(structured_csv))
    
    # Save labeled sequences
    labeled_csv = output_path / "ait_sequence_labeled.csv"
    ait_sequence_builder.save_labeled_sequences(labeled_sequences_df, str(labeled_csv))
    
    print(f"Created {len(labeled_sequences_df):,} AIT-LDS sequences with labels")

    # Step 5: Text Conversion for Model Input
    print("\n" + "="*70)
    print("STEP 5: TEXT CONVERSION")
    print("="*70)
    
    text_converter = TextConverter(config, dataset_type='ait')
    text_sequences_df = text_converter.convert_sequences_to_text(labeled_sequences_df)
    print(f"Converted {len(text_sequences_df):,} sequences to text format")

    # Step 6: Train/Test Split
    print("\n" + "="*70)
    print("STEP 6: TRAIN/TEST SPLIT")
    print("="*70)
    
    data_splitter = DataSplitter(config, dataset_type='ait')
    train_df, test_df = data_splitter.split_data(text_sequences_df)
    print(f"Train set size: {len(train_df):,} samples")
    print(f"Test set size: {len(test_df):,} samples")
    print(f"Train and test sets saved to: {output_path}")

    # Final summary
    print("\n" + "="*70)
    print("AIT-LDS PREPROCESSING COMPLETE!")
    print("="*70)
    
    print(f"\nGenerated files:")
    files = [
        ("Structured logs", structured_csv),
        ("Sequences", sequences_csv),
        ("Labeled sequences", labeled_csv),
        ("Text sequences", output_path / "ait_text.csv"),
        ("Train set", output_path / "ait_train.csv"),
        ("Test set", output_path / "ait_test.csv"),
        ("Template mapping", template_mapping_file)
    ]
    
    for name, path in files:
        if path.exists():
            size = path.stat().st_size / (1024 * 1024)  # MB
            print(f"  ✓ {name}: {path.name} ({size:.2f} MB)")
        else:
            print(f"  ✗ {name}: {path.name} (not found)")
    
    print(f"\nDataset: {dataset_name}")
    print(f"Total log entries: {len(structured_df):,}")
    print(f"Unique templates: {structured_df['EventTemplate'].nunique():,}")
    print(f"Total sequences: {len(labeled_sequences_df):,}")
    
    if 'Label' in labeled_sequences_df.columns:
        attack_count = labeled_sequences_df['Label'].sum()
        normal_count = len(labeled_sequences_df) - attack_count
        attack_rate = (attack_count / len(labeled_sequences_df)) * 100
        
        print(f"Normal sequences: {normal_count:,} ({100-attack_rate:.2f}%)")
        print(f"Attack sequences: {attack_count:,} ({attack_rate:.2f}%)")
    
    print("\n" + "="*70)
    print("Ready for AIT-LDS benchmarking!")
    print("Run: python scripts/benchmark_ait.py")
    print("="*70)


if __name__ == "__main__":
    main()
