"""
AIT-LDS Preprocessing Pipeline

Complete preprocessing pipeline for AIT Log Data Sets.
Handles multi-host logs, attack labels, and sequence building.
"""

import os
import sys
import json
import gc
from pathlib import Path
import yaml
import pandas as pd

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

from parsers.drain import Drain as DrainParser
from preprocessing.ait_parser import AITLogParser
from preprocessing.template_mapper import TemplateMapper
from preprocessing.ait_sequence_builder import AITSequenceBuilder
from preprocessing.data_splitter import DataSplitter
from preprocessing.text_converter import TextConverter


def parse_log_chunk_direct(chunk_data: list, host_name: str, log_type: str, start_id: int) -> pd.DataFrame:
    """
    Parse a chunk of log data directly without using AITLogParser methods.
    This avoids module reloading issues in Colab.
    """
    parsed_logs = []
    
    # Basic log patterns for common log types
    log_patterns = {
        'apache_access': r'^(\d+\.\d+\.\d+\.\d+) - - \[([^\]]+)\] "([^"]+)" (\d+) (\d+) "([^"]*)" "([^"]*)"$',
        'apache_error': r'^\[([^\]]+)\] \[([^\]]+)\] \[client ([^\]]+)\] ([^:]+): (.+)$',
        'auth': r'^(\w{3} \d{1,2} \d{2}:\d{2}:\d{2}) (\S+) (\w+)(\[[\d]+\])?: (.+)$',
        'audit': r'^type=(\w+) msg=audit\(([^)]+)\): (.+)$',
        'syslog': r'^(\w{3} \d{1,2} \d{2}:\d{2}:\d{2}) (\S+) (\S+): (.+)$',
    }
    
    for i, line in enumerate(chunk_data):
        log_id = start_id + i
        
        # Create basic entry
        entry = {
            'LogId': log_id,
            'Host': host_name,
            'LogType': log_type,
            'Content': line,
            'Timestamp': '',
            'Level': '',
            'Message': line,
            'Parsed': False
        }
        
        # Try to parse based on log type
        if log_type in log_patterns:
            import re
            pattern = log_patterns[log_type]
            match = re.match(pattern, line)
            
            if match:
                entry['Parsed'] = True
                
                if log_type == 'apache_access':
                    entry.update({
                        'Timestamp': match.group(2),
                        'Source': match.group(1),
                        'Method': match.group(3).split()[0] if ' ' in match.group(3) else match.group(3),
                        'Status': match.group(4),
                        'Size': match.group(5),
                        'UserAgent': match.group(7)
                    })
                elif log_type == 'apache_error':
                    entry.update({
                        'Timestamp': match.group(1),
                        'Level': match.group(2),
                        'Source': match.group(3),
                        'ErrorType': match.group(4),
                        'Message': match.group(5)
                    })
                elif log_type == 'auth':
                    entry.update({
                        'Timestamp': match.group(1),
                        'Source': match.group(2),
                        'Process': match.group(3),
                        'Message': match.group(5)
                    })
                elif log_type == 'audit':
                    entry.update({
                        'EventType': match.group(1),
                        'Timestamp': match.group(2),
                        'Details': match.group(3)
                    })
                elif log_type == 'syslog':
                    entry.update({
                        'Timestamp': match.group(1),
                        'Source': match.group(2),
                        'Process': match.group(3),
                        'Message': match.group(4)
                    })
        
        parsed_logs.append(entry)
    
    return pd.DataFrame(parsed_logs)


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
    
    # Check if dataset is in subdirectory or directly in base_path
    dataset_path = base_path / dataset_name
    if not dataset_path.exists():
        # Try direct path (dataset extracted directly)
        dataset_path = base_path
        print(f"Dataset not found in subdirectory, trying direct path: {dataset_path}")
    
    # Additional check: if we're using base_path, make sure it has the required structure
    if dataset_path == base_path:
        # Check if the required directories exist directly in base_path
        gather_dir = dataset_path / 'gather'
        if gather_dir.exists():
            print(f"Using direct extraction path: {dataset_path}")
        else:
            # Try the subdirectory approach
            dataset_path = base_path / dataset_name
            print(f"Falling back to subdirectory path: {dataset_path}")
    
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
    
    # Check for required directories
    gather_dir = dataset_path / 'gather'
    if not gather_dir.exists():
        print(f"Error: gather directory not found: {gather_dir}")
        print("Please check if the dataset was extracted correctly")
        sys.exit(1)

    # Step 1: AIT Log Parsing (multi-host structure) - CHUNKED PROCESSING
    print("\n" + "="*70)
    print("STEP 1: AIT-LDS LOG PARSING (MEMORY-EFFICIENT)")
    print("="*70)
    
    ait_parser = AITLogParser(config)
    # Override the dataset_path if we detected a different structure
    ait_parser.dataset_path = dataset_path
    ait_parser.dataset_yml = dataset_path / 'dataset.yml'
    ait_parser.dataset_info = ait_parser._load_dataset_info()
    
    # Process logs in chunks to avoid memory overflow
    chunk_size = 50000  # Process 50k logs at a time
    structured_csv = output_path / output_config['structured_log']
    
    print(f"Processing logs in chunks of {chunk_size:,} entries...")
    
    # Initialize counters
    total_processed = 0
    first_chunk = True
    
    # Get all hosts
    gather_dir = dataset_path / 'gather'
    hosts = [d for d in gather_dir.iterdir() if d.is_dir()]
    print(f"Found {len(hosts)} hosts: {[h.name for h in hosts]}")
    
    for host in hosts:
        print(f"\nProcessing host: {host.name}")
        
        # Get all log files for this host
        logs_dir = host / 'logs'
        if not logs_dir.exists():
            continue
            
        log_files = []
        for log_type_dir in logs_dir.iterdir():
            if log_type_dir.is_dir():
                for log_file in log_type_dir.iterdir():
                    if log_file.is_file():
                        log_files.append(log_file)
        
        print(f"  Found {len(log_files)} log files")
        
        # Process files in chunks
        for log_file in log_files:
            log_type = log_file.parent.name
            
            print(f"    Processing {log_type}: {log_file.name}")
            
            # Read file in chunks
            chunk_data = []
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if line:
                        chunk_data.append(line)
                        
                        # Process chunk when it reaches chunk_size
                        if len(chunk_data) >= chunk_size:
                            # Parse chunk directly without using parse_log_chunk method
                            chunk_df = parse_log_chunk_direct(chunk_data, host.name, log_type, total_processed)
                            
                            if not chunk_df.empty:
                                # Save chunk to CSV (append mode)
                                mode = 'w' if first_chunk else 'a'
                                header = first_chunk
                                chunk_df.to_csv(structured_csv, mode=mode, header=header, index=False)
                                first_chunk = False
                                
                                total_processed += len(chunk_df)
                                print(f"      Processed {len(chunk_df):,} entries (Total: {total_processed:,})")
                            
                            # Clear chunk data to free memory
                            chunk_data = []
                            gc.collect()  # Force garbage collection
            
            # Process remaining data in chunk
            if chunk_data:
                chunk_df = parse_log_chunk_direct(chunk_data, host.name, log_type, total_processed)
                
                if not chunk_df.empty:
                    mode = 'w' if first_chunk else 'a'
                    header = first_chunk
                    chunk_df.to_csv(structured_csv, mode=mode, header=header, index=False)
                    first_chunk = False
                    
                    total_processed += len(chunk_df)
                    print(f"      Processed {len(chunk_df):,} entries (Total: {total_processed:,})")
    
    if total_processed == 0:
        print("Error: No logs were parsed from AIT-LDS dataset")
        sys.exit(1)
    
    print(f"\nParsed {total_processed:,} AIT-LDS log entries total")

    # Step 2: Event Template Extraction (using Drain) - CHUNKED PROCESSING
    print("\n" + "="*70)
    print("STEP 2: EVENT TEMPLATE EXTRACTION (DRAIN) - MEMORY-EFFICIENT")
    print("="*70)
    
    # Process Drain in chunks to avoid memory overflow
    drain_chunk_size = 100000  # Process 100k logs at a time for Drain
    
    # Create Drain parser with proper configuration
    drain_parser = DrainParser(
        depth=drain_config['depth'],
        st=drain_config['st'],
        rex=drain_config.get('rex', [])
    )
    
    print(f"Processing Drain extraction in chunks of {drain_chunk_size:,} entries...")
    
    # Read structured CSV in chunks and process with Drain
    drain_results = []
    chunk_num = 0
    
    # Read the structured CSV in chunks
    chunk_iter = pd.read_csv(structured_csv, chunksize=drain_chunk_size)
    
    for chunk_df in chunk_iter:
        chunk_num += 1
        print(f"  Processing Drain chunk {chunk_num} ({len(chunk_df):,} entries)...")
        
        # Extract content for Drain
        content_lines = chunk_df['Content'].tolist()
        
        # Process with Drain
        event_ids = []
        event_templates = []
        
        for log_id, line in enumerate(content_lines):
            if line and str(line).strip():
                event_id, template = drain_parser.parse(log_id, line)
                event_ids.append(event_id)
                event_templates.append(template)
        
        # Add Drain results to chunk
        chunk_df['EventId'] = event_ids[:len(chunk_df)]
        chunk_df['EventTemplate'] = event_templates[:len(chunk_df)]
        
        # Save updated chunk
        mode = 'w' if chunk_num == 1 else 'a'
        header = chunk_num == 1
        chunk_df.to_csv(structured_csv, mode=mode, header=header, index=False)
        
        print(f"    Processed {len(chunk_df):,} entries with Drain")
        gc.collect()  # Force garbage collection after each chunk
    
    # Count unique templates
    print("Counting unique templates...")
    unique_templates = set()
    chunk_iter = pd.read_csv(structured_csv, chunksize=drain_chunk_size, usecols=['EventTemplate'])
    
    for chunk_df in chunk_iter:
        unique_templates.update(chunk_df['EventTemplate'].unique())
    
    print(f"Extracted {len(unique_templates):,} unique event templates")

    # Step 3: Template Mapping - CHUNKED PROCESSING
    print("\n" + "="*70)
    print("STEP 3: TEMPLATE MAPPING (MEMORY-EFFICIENT)")
    print("="*70)
    
    template_mapper = TemplateMapper()
    
    # Create templates DataFrame in chunks to avoid memory overflow
    print("Creating template mapping from structured logs...")
    
    template_counts = {}
    chunk_iter = pd.read_csv(structured_csv, chunksize=drain_chunk_size, usecols=['EventId', 'EventTemplate'])
    
    chunk_num = 0
    for chunk_df in chunk_iter:
        chunk_num += 1
        print(f"  Processing template mapping chunk {chunk_num}...")
        
        # Count templates in this chunk
        chunk_templates = chunk_df.groupby(['EventId', 'EventTemplate']).size().reset_index(name='Occurrences')
        
        # Merge with existing counts
        for _, row in chunk_templates.iterrows():
            key = (row['EventId'], row['EventTemplate'])
            template_counts[key] = template_counts.get(key, 0) + row['Occurrences']
    
    # Create final templates DataFrame
    templates_data = []
    for (event_id, template), count in template_counts.items():
        templates_data.append({
            'EventId': event_id,
            'EventTemplate': template,
            'Occurrences': count
        })
    
    templates_df = pd.DataFrame(templates_data)
    templates_df = templates_df.sort_values('Occurrences', ascending=False)
    
    # Save templates CSV
    templates_csv = output_path / output_config['templates']
    templates_df.to_csv(templates_csv, index=False)
    
    # Create mapping
    template_mapping_file = output_path / output_config['template_mapping']
    template_mapper.create_mapping(str(templates_csv), str(template_mapping_file))
    print(f"Template mapping saved")

    # Step 4: Sequence Creation & Label Assignment
    print("\n" + "="*70)
    print("STEP 4: SEQUENCE CREATION & LABEL ASSIGNMENT")
    print("="*70)
    
    ait_sequence_builder = AITSequenceBuilder(config)
    
    # Load template mapping
    template_mapping_file = output_path / output_config['template_mapping']
    if template_mapping_file.exists():
        with open(template_mapping_file, 'r') as f:
            template_mapping = json.load(f)
        
        # Use the event_id_to_number mapping directly
        if 'event_id_to_number' in template_mapping:
            event_id_mapping = {int(k): int(v) for k, v in template_mapping['event_id_to_number'].items()}
        else:
            # Fallback: create mapping from template content
            event_id_mapping = {}
            for template_id, template_content in template_mapping.items():
                # Find EventId for this template
                matching_rows = structured_df[structured_df['EventTemplate'] == template_content]
                if not matching_rows.empty:
                    event_id = matching_rows['EventId'].iloc[0]
                    event_id_mapping[int(event_id)] = int(template_id)
        
    else:
        print("Warning: Template mapping file not found, creating simple mapping")
        # Create simple mapping from EventId to template number
        unique_templates = structured_df['EventTemplate'].unique()
        event_id_mapping = {}
        for i, template in enumerate(unique_templates):
            matching_rows = structured_df[structured_df['EventTemplate'] == template]
            if not matching_rows.empty:
                event_id = matching_rows['EventId'].iloc[0]
                event_id_mapping[int(event_id)] = i + 1  # Start from 1, not 0
        
    
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
    
    # Save text sequences
    text_csv = output_path / "ait_text.csv"
    text_sequences_df.to_csv(text_csv, index=False)
    print(f"Text sequences saved to: {text_csv}")

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
            print(f"  [OK] {name}: {path.name} ({size:.2f} MB)")
        else:
            print(f"  [MISSING] {name}: {path.name} (not found)")
    
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
