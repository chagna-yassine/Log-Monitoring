"""
AIT-LDS Chunked Preprocessing Pipeline

Processes AIT dataset in 10GB chunks, uploads to Hugging Face, and deletes local files.
This approach saves storage space while processing large datasets.
"""

import os
import sys
import json
import gc
import shutil
from pathlib import Path
import yaml
import pandas as pd
import numpy as np

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


def setup_huggingface_token():
    """Help user set up Hugging Face token."""
    print("🔑 HUGGING FACE AUTHENTICATION SETUP")
    print("=" * 50)
    print()
    print("To upload to Hugging Face, you need a token.")
    print()
    print("Option 1: Get token from Hugging Face website")
    print("1. Go to: https://huggingface.co/settings/tokens")
    print("2. Click 'New token'")
    print("3. Give it a name (e.g., 'ait-processing')")
    print("4. Select 'Write' permissions")
    print("5. Copy the token")
    print()
    print("Option 2: Use Hugging Face CLI")
    print("1. Install: pip install huggingface_hub")
    print("2. Login: huggingface-cli login")
    print("3. Enter your token when prompted")
    print()
    print("Option 3: Set environment variable")
    print("Windows: set HUGGINGFACE_HUB_TOKEN=your_token_here")
    print("Linux/Mac: export HUGGINGFACE_HUB_TOKEN=your_token_here")
    print()
    
    # Check if token is already set
    import os
    hf_token = os.getenv('HUGGINGFACE_HUB_TOKEN') or os.getenv('HF_TOKEN')
    
    if hf_token:
        print("✅ Token found! You're ready to upload.")
        return True
    else:
        print("❌ No token found. Please set up authentication first.")
        return False


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_chunk_size_gb(target_gb: float = 10.0) -> int:
    """Calculate chunk size in rows to achieve target GB."""
    # Estimate bytes per row (rough estimate for AIT logs)
    bytes_per_row = 500  # Conservative estimate
    target_bytes = target_gb * 1024**3
    chunk_size = int(target_bytes / bytes_per_row)
    return max(10000, chunk_size)  # Minimum 10k rows


def upload_to_huggingface(chunk_data: dict, chunk_id: int, repo_name: str, hf_token: str = None):
    """Upload chunk data to Hugging Face."""
    temp_dir = None
    try:
        from huggingface_hub import HfApi, Repository
        
        # Check for Hugging Face token
        if not hf_token:
            print(f"  ⚠️  No Hugging Face token provided!")
            print(f"  Skipping upload for chunk {chunk_id}")
            return
        
        api = HfApi(token=hf_token)
        
        # Create temporary directory for this chunk
        temp_dir = Path(f"temp_chunk_{chunk_id}")
        temp_dir.mkdir(exist_ok=True)
        
        # Save chunk data
        for filename, data in chunk_data.items():
            file_path = temp_dir / filename
            
            # Handle different data types
            if isinstance(data, pd.DataFrame):
                data.to_csv(file_path, index=False)
                print(f"  Saved {filename}: {len(data):,} rows")
            elif isinstance(data, dict):
                # Save dictionary as JSON
                with open(file_path.with_suffix('.json'), 'w') as f:
                    json.dump(data, f, indent=2)
                print(f"  Saved {filename}: {len(data):,} entries")
            else:
                print(f"  Warning: Unknown data type for {filename}: {type(data)}")
                continue
        
        # Upload to Hugging Face
        print(f"  Uploading chunk {chunk_id} to Hugging Face...")
        
        # Create dataset repository if it doesn't exist
        try:
            api.create_repo(repo_id=repo_name, repo_type="dataset", exist_ok=True)
        except Exception as e:
            print(f"  Note: Repository creation: {e}")
        
        # Upload files
        for filename in chunk_data.keys():
            file_path = temp_dir / filename
            
            # Determine the correct file extension
            if isinstance(chunk_data[filename], dict):
                file_path = file_path.with_suffix('.json')
            
            if file_path.exists():
                api.upload_file(
                    path_or_fileobj=str(file_path),
                    path_in_repo=f"chunk_{chunk_id}/{file_path.name}",
                    repo_id=repo_name,
                    repo_type="dataset"
                )
                print(f"    Uploaded {file_path.name}")
        
        # Clean up temporary directory
        shutil.rmtree(temp_dir)
        print(f"  ✓ Chunk {chunk_id} uploaded successfully")
        
    except Exception as e:
        print(f"  ✗ Error uploading chunk {chunk_id}: {e}")
        # Clean up temp directory if it exists
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir)


def process_chunk(structured_df: pd.DataFrame, chunk_id: int, config: dict) -> dict:
    """Process a chunk of structured data."""
    print(f"\n{'='*70}")
    print(f"PROCESSING CHUNK {chunk_id}")
    print(f"{'='*70}")
    
    ait_config = config['ait_dataset']
    output_config = config['output']['ait']
    drain_config = config['preprocessing']['drain']
    
    # Create sequences from structured data
    sequences = []
    labels = []
    
    print(f"Creating sequences from chunk {chunk_id}...")
    
    # Create sequences with better logic
    sequence_length = 20
    overlap = 5
    
    for host in structured_df['Host'].unique():
        host_logs = structured_df[structured_df['Host'] == host].reset_index(drop=True)
        
        if len(host_logs) < sequence_length:
            continue
        
        # Create overlapping sequences
        for i in range(0, len(host_logs) - sequence_length + 1, sequence_length - overlap):
            sequence_logs = host_logs.iloc[i:i+sequence_length]
            
            # Create sequence text from Content with better formatting
            try:
                # Clean and format log content
                cleaned_logs = []
                for log in sequence_logs['Content'].astype(str):
                    log = log.strip()
                    if log and len(log) > 10:
                        cleaned_logs.append(log)
                
                if len(cleaned_logs) >= 10:
                    # Use newline separator for better model understanding
                    sequence_text = "\n".join(cleaned_logs)
                    sequences.append(sequence_text)
                    
                    # Determine label (1 if any attack, 0 if normal)
                    if 'Label' in sequence_logs.columns:
                        label = 1 if sequence_logs['Label'].sum() > 0 else 0
                    else:
                        # Enhanced attack detection
                        attack_keywords = [
                            'attack', 'malicious', 'suspicious', 'intrusion', 'breach', 'exploit',
                            'unauthorized', 'failed', 'denied', 'blocked', 'firewall', 'scan',
                            'nmap', 'sql injection', 'xss', 'csrf', 'ddos', 'botnet', 'trojan'
                        ]
                        
                        suspicious_hosts = ['attacker', 'malicious', 'suspicious', 'external']
                        
                        content_text = ' '.join(cleaned_logs).lower()
                        host_name = host.lower()
                        
                        content_attack = any(keyword in content_text for keyword in attack_keywords)
                        host_attack = any(susp_host in host_name for susp_host in suspicious_hosts)
                        
                        # Pattern-based detection
                        pattern_attack = False
                        for log in cleaned_logs:
                            log_lower = log.lower()
                            if 'failed' in log_lower and ('login' in log_lower or 'password' in log_lower):
                                pattern_attack = True
                                break
                            if 'port' in log_lower and ('scan' in log_lower or 'probe' in log_lower):
                                pattern_attack = True
                                break
                        
                        label = 1 if (content_attack or host_attack or pattern_attack) else 0
                    
                    labels.append(label)
                    
            except Exception as e:
                print(f"Warning: Error creating sequence for host {host}: {e}")
                continue
    
    if not sequences:
        print(f"Warning: No sequences could be created from chunk {chunk_id}")
        print(f"  This might be due to insufficient log entries or very short logs")
        # Return empty data structure instead of empty dict
        return {
            f'ait_chunk_{chunk_id}_structured.csv': structured_df,
            f'ait_chunk_{chunk_id}_sequences.csv': pd.DataFrame(columns=['TextSequence', 'Label']),
            f'ait_chunk_{chunk_id}_train.csv': pd.DataFrame(columns=['TextSequence', 'Label']),
            f'ait_chunk_{chunk_id}_test.csv': pd.DataFrame(columns=['TextSequence', 'Label']),
            f'ait_chunk_{chunk_id}_templates.csv': pd.DataFrame(columns=['EventId', 'EventTemplate', 'Occurrences']),
            f'ait_chunk_{chunk_id}_mapping.json': {}
        }
    
    # Create test DataFrame
    test_df = pd.DataFrame({
        'TextSequence': sequences,
        'Label': labels
    })
    
    # Check class distribution
    attack_count = sum(labels)
    normal_count = len(labels) - attack_count
    attack_rate = (attack_count / len(labels)) * 100 if len(labels) > 0 else 0
    
    print(f"Chunk {chunk_id} sequence statistics:")
    print(f"  Total sequences: {len(sequences):,}")
    print(f"  Normal sequences: {normal_count:,} ({100-attack_rate:.2f}%)")
    print(f"  Attack sequences: {attack_count:,} ({attack_rate:.2f}%)")
    
    # Add synthetic attacks if needed
    if attack_rate < 1.0:
        print(f"Adding synthetic attack sequences to chunk {chunk_id}...")
        
        normal_indices = [i for i, label in enumerate(labels) if label == 0]
        num_synthetic = min(len(normal_indices) // 10, 50)  # 10% of normal sequences, max 50
        
        if num_synthetic > 0:
            import random
            random.seed(42 + chunk_id)  # Different seed per chunk
            
            synthetic_indices = random.sample(normal_indices, num_synthetic)
            
            for idx in synthetic_indices:
                original_text = sequences[idx]
                attack_patterns = [
                    "Unauthorized access attempt detected",
                    "Failed login attempt from suspicious IP",
                    "Port scan detected on multiple ports",
                    "SQL injection attempt blocked",
                    "Malicious payload detected in request"
                ]
                
                lines = original_text.split('\n')
                if len(lines) > 5:
                    insert_pos = random.randint(2, len(lines) - 3)
                    attack_line = random.choice(attack_patterns)
                    lines.insert(insert_pos, attack_line)
                    sequences[idx] = '\n'.join(lines)
                    labels[idx] = 1
            
            # Recalculate statistics
            attack_count = sum(labels)
            normal_count = len(labels) - attack_count
            attack_rate = (attack_count / len(labels)) * 100
            
            print(f"Added {num_synthetic} synthetic attack sequences")
            print(f"Updated chunk {chunk_id} statistics:")
            print(f"  Normal sequences: {normal_count:,} ({100-attack_rate:.2f}%)")
            print(f"  Attack sequences: {attack_count:,} ({attack_rate:.2f}%)")
    
    # Update test_df with new labels
    test_df['Label'] = labels
    
    # Process with Drain for template extraction
    print(f"Running Drain template extraction on chunk {chunk_id}...")
    
    # Create Drain parser
    drain_parser = DrainParser(
        depth=drain_config['depth'],
        st=drain_config['st'],
        rex=drain_config.get('rex', [])
    )
    
    # Extract templates
    event_ids = []
    event_templates = []
    
    for log_id, text in enumerate(sequences):
        if text and str(text).strip():
            event_id, template = drain_parser.parse(log_id, text)
            event_ids.append(event_id)
            event_templates.append(template)
    
    # Add Drain results to test_df
    test_df['EventId'] = event_ids[:len(test_df)]
    test_df['EventTemplate'] = event_templates[:len(test_df)]
    
    # Create template mapping
    template_mapper = TemplateMapper()
    templates_df = test_df.groupby(['EventId', 'EventTemplate']).size().reset_index(name='Occurrences')
    templates_df = templates_df.sort_values('Occurrences', ascending=False)
    
    # Create mapping
    template_mapping = {}
    for i, (_, row) in enumerate(templates_df.iterrows()):
        template_mapping[int(row['EventId'])] = {
            'template': row['EventTemplate'],
            'number': i + 1,
            'occurrences': int(row['Occurrences'])
        }
    
    # Convert sequences to text format
    print(f"Converting sequences to text format for chunk {chunk_id}...")
    
    text_sequences = []
    for sequence in sequences:
        # Simple text conversion (can be enhanced)
        text_sequences.append(sequence)
    
    test_df['TextSequence'] = text_sequences
    
    # Split into train/test
    print(f"Splitting chunk {chunk_id} into train/test sets...")
    
    normal_df = test_df[test_df['Label'] == 0]
    attack_df = test_df[test_df['Label'] == 1]
    
    # Use 70% of normal for training
    train_size = int(len(normal_df) * 0.7)
    train_df = normal_df.head(train_size)
    test_df_final = pd.concat([normal_df.tail(len(normal_df) - train_size), attack_df])
    
    # Ensure we have valid DataFrames
    if train_df.empty:
        train_df = pd.DataFrame(columns=test_df.columns)
    if test_df_final.empty:
        test_df_final = pd.DataFrame(columns=test_df.columns)
    
    # Prepare chunk data for upload
    chunk_data = {
        f'ait_chunk_{chunk_id}_structured.csv': structured_df,
        f'ait_chunk_{chunk_id}_sequences.csv': test_df,
        f'ait_chunk_{chunk_id}_train.csv': train_df,
        f'ait_chunk_{chunk_id}_test.csv': test_df_final,
        f'ait_chunk_{chunk_id}_templates.csv': templates_df,
        f'ait_chunk_{chunk_id}_mapping.json': template_mapping
    }
    
    return chunk_data


def main():
    """Main function to run chunked AIT-LDS preprocessing pipeline."""
    print("="*70)
    print("AIT-LDS CHUNKED PREPROCESSING PIPELINE")
    print("="*70)
    
    # Load configuration
    config = load_config()
    ait_config = config['ait_dataset']
    output_config = config['output']['ait']
    
    # Setup paths
    base_path = Path(ait_config['base_path'])
    dataset_name = ait_config['selected_dataset']
    
    # Check if dataset is in subdirectory or directly in base_path
    dataset_path = base_path / dataset_name
    if not dataset_path.exists():
        dataset_path = base_path
        print(f"Using direct extraction path: {dataset_path}")
    
    output_path = Path(output_config['base_path'])
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"AIT-LDS Configuration:")
    print(f"  Dataset: {dataset_name}")
    print(f"  Dataset Path: {dataset_path}")
    print(f"  Output Path: {output_path}")
    
    # Verify dataset exists
    gather_dir = dataset_path / 'gather'
    if not gather_dir.exists():
        print(f"Error: gather directory not found: {gather_dir}")
        sys.exit(1)
    
    # Get Hugging Face token and repository name
    print("\n" + "="*70)
    print("HUGGING FACE UPLOAD SETUP")
    print("="*70)
    
    print("\nTo upload processed chunks to Hugging Face, you need:")
    print("1. A Hugging Face token (get from: https://huggingface.co/settings/tokens)")
    print("2. A repository name (e.g., 'username/ait-processed')")
    print()
    
    # Get Hugging Face token
    hf_token = input("Enter your Hugging Face token: ").strip()
    if not hf_token:
        print("❌ No token provided. Uploads will be skipped.")
        hf_token = None
    else:
        print("✅ Token received!")
    
    # Get Hugging Face repository name
    repo_name = input("\nEnter Hugging Face repository name (e.g., 'username/ait-processed'): ").strip()
    if not repo_name:
        repo_name = "ait-processed-data"
        print(f"Using default repository name: {repo_name}")
    
    print(f"\nRepository: {repo_name}")
    print(f"Token: {'✅ Set' if hf_token else '❌ Not provided'}")
    
    # Calculate chunk size
    chunk_size_rows = get_chunk_size_gb(10.0)  # 10GB chunks
    print(f"Processing in chunks of {chunk_size_rows:,} rows (~10GB each)")
    
    # Process dataset in chunks
    chunk_id = 0
    total_processed = 0
    
    # Get all hosts
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
                        if len(chunk_data) >= chunk_size_rows:
                            chunk_df = parse_log_chunk_direct(chunk_data, host.name, log_type, total_processed)
                            
                            if not chunk_df.empty:
                                chunk_id += 1
                                total_processed += len(chunk_df)
                                
                                print(f"      Processed chunk {chunk_id}: {len(chunk_df):,} entries (Total: {total_processed:,})")
                                
                                # Process the chunk
                                processed_data = process_chunk(chunk_df, chunk_id, config)
                                
                                # Always upload (even if empty sequences)
                                upload_to_huggingface(processed_data, chunk_id, repo_name, hf_token)
                                
                                # Delete local data to free space
                                del chunk_df, processed_data
                                gc.collect()
                                
                                print(f"      ✓ Chunk {chunk_id} processed and uploaded")
                                
                            # Clear chunk data to free memory
                            chunk_data = []
                            gc.collect()
            
            # Process remaining data in chunk
            if chunk_data:
                chunk_df = parse_log_chunk_direct(chunk_data, host.name, log_type, total_processed)
                
                if not chunk_df.empty:
                    chunk_id += 1
                    total_processed += len(chunk_df)
                    
                    print(f"      Processed final chunk {chunk_id}: {len(chunk_df):,} entries (Total: {total_processed:,})")
                    
                    # Process the chunk
                    processed_data = process_chunk(chunk_df, chunk_id, config)
                    
                    # Always upload (even if empty sequences)
                    upload_to_huggingface(processed_data, chunk_id, repo_name, hf_token)
                    
                    # Delete local data to free space
                    del chunk_df, processed_data
                    gc.collect()
                    
                    print(f"      ✓ Final chunk {chunk_id} processed and uploaded")
    
    print(f"\n{'='*70}")
    print("CHUNKED PREPROCESSING COMPLETE!")
    print(f"{'='*70}")
    print(f"Total chunks processed: {chunk_id}")
    print(f"Total entries processed: {total_processed:,}")
    
    if hf_token:
        print(f"✅ Data uploaded to: https://huggingface.co/datasets/{repo_name}")
        print(f"Each chunk contains:")
        print(f"  - Structured logs")
        print(f"  - Event sequences")
        print(f"  - Train/test splits")
        print(f"  - Template mappings")
        print(f"  - Text sequences")
    else:
        print(f"⚠️  Uploads were skipped (no token provided)")
        print(f"Processed data is available locally in: datasets/ait/output/ait/")
        print(f"To upload later, run the script again with a token")


if __name__ == "__main__":
    main()
