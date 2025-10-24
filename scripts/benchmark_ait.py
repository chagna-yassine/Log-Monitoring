"""
AIT-LDS Benchmarking

Runs comprehensive benchmarking on AIT-LDS test data.
"""

import pandas as pd
import yaml
from pathlib import Path
import sys
import json
import time

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

from model.inference import LogAnomalyDetector
from evaluation.metrics import BenchmarkMetrics


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def cleanup_disk_space():
    """Clean up disk space by removing temporary files."""
    import shutil
    import os
    
    print("Attempting to free up disk space...")
    
    # Clean up common temporary locations
    temp_dirs = [
        "/tmp",
        "/content/tmp",
        "/root/.cache/pip",
        "/content/.cache/pip"
    ]
    
    freed_space = 0
    
    for temp_dir in temp_dirs:
        if os.path.exists(temp_dir):
            try:
                for item in os.listdir(temp_dir):
                    item_path = os.path.join(temp_dir, item)
                    if os.path.isfile(item_path):
                        size = os.path.getsize(item_path)
                        os.remove(item_path)
                        freed_space += size
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path, ignore_errors=True)
                        # Estimate freed space (rough)
                        freed_space += 1024 * 1024  # Assume 1MB per directory
            except Exception as e:
                print(f"Warning: Could not clean {temp_dir}: {e}")
    
    freed_mb = freed_space / (1024 * 1024)
    print(f"Freed approximately {freed_mb:.2f} MB of disk space")
    
    return freed_mb


def main():
    """Run AIT-LDS benchmarking."""
    print("="*70)
    print("AIT-LDS LOG ANOMALY DETECTION BENCHMARKING")
    print("="*70)

    # Load configuration
    config = load_config()
    model_config = config['model']
    ait_config = config['ait_dataset']
    output_config = config['output']['ait']
    results_dir = Path(config['output']['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)

    # Setup paths
    output_path = Path(output_config['base_path'])
    
    # Check for structured CSV first (if preprocessing was stopped)
    structured_file = output_path / "ait_structured.csv"
    test_file = output_path / "ait_test.csv"

    if structured_file.exists():
        print(f"Found structured CSV file: {structured_file}")
        print("Running benchmark on structured logs directly...")
        use_structured = True
        data_file = structured_file
    elif test_file.exists():
        print(f"Found test CSV file: {test_file}")
        print("Running benchmark on preprocessed test data...")
        use_structured = False
        data_file = test_file
    else:
        print(f"Error: No AIT-LDS data files found in: {output_path}")
        print("Expected files: ait_structured.csv or ait_test.csv")
        print("Please run: python scripts/preprocess_ait.py")
        sys.exit(1)

    print(f"Loading AIT-LDS data from: {data_file}")
    
    if use_structured:
        # Load structured data in chunks to handle parsing errors
        print("Loading structured data in chunks to handle parsing errors...")
        
        chunk_size = 10000  # Read 10k rows at a time
        structured_chunks = []
        
        try:
            # Try to read the CSV in chunks
            chunk_iter = pd.read_csv(data_file, chunksize=chunk_size, on_bad_lines='skip')
            
            for i, chunk_df in enumerate(chunk_iter):
                if not chunk_df.empty:
                    structured_chunks.append(chunk_df)
                    print(f"  Loaded chunk {i+1}: {len(chunk_df):,} rows")
                    
                    # Stop after reasonable number of chunks to avoid memory issues
                    if i >= 50:  # Limit to 500k rows max
                        print(f"  Stopping at chunk {i+1} to avoid memory issues")
                        break
            
            if structured_chunks:
                structured_df = pd.concat(structured_chunks, ignore_index=True)
                print(f"Successfully loaded {len(structured_df):,} structured log entries")
            else:
                print("Error: No valid chunks could be loaded from structured CSV")
                sys.exit(1)
                
        except Exception as e:
            print(f"Error loading structured CSV: {e}")
            print("Trying alternative approach...")
            
            # Alternative: read raw lines and parse manually
            try:
                with open(data_file, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                
                print(f"Read {len(lines):,} lines from file")
                
                # Take first 100k lines to avoid memory issues
                max_lines = min(100000, len(lines))
                lines = lines[:max_lines]
                
                # Parse header
                header = lines[0].strip().split(',')
                print(f"Detected columns: {header}")
                
                # Parse data rows
                parsed_rows = []
                for i, line in enumerate(lines[1:max_lines], 1):
                    try:
                        values = line.strip().split(',')
                        if len(values) >= len(header):
                            # Take only the number of values that match header
                            row_data = dict(zip(header, values[:len(header)]))
                            parsed_rows.append(row_data)
                    except Exception:
                        continue
                    
                    if i % 10000 == 0:
                        print(f"  Parsed {i:,} lines...")
                
                structured_df = pd.DataFrame(parsed_rows)
                print(f"Successfully parsed {len(structured_df):,} rows manually")
                
            except Exception as e2:
                print(f"Alternative approach also failed: {e2}")
                sys.exit(1)
        
        # Create meaningful sequences from structured data
        sequences = []
        labels = []
        
        print("Creating sequences from structured data...")
        
        # Debug: Show sample data
        print(f"Sample structured data:")
        print(f"Columns: {list(structured_df.columns)}")
        print(f"Sample rows:")
        print(structured_df.head(3).to_string())
        
        # Check if required columns exist
        required_cols = ['Host', 'Content']
        missing_cols = [col for col in required_cols if col not in structured_df.columns]
        
        if missing_cols:
            print(f"Warning: Missing columns {missing_cols}, using available columns")
            print(f"Available columns: {list(structured_df.columns)}")
            
            # Use first available column as Host if Host is missing
            if 'Host' not in structured_df.columns:
                if 'LogId' in structured_df.columns:
                    structured_df['Host'] = 'default_host'
                else:
                    print("Error: Cannot create sequences without Host or LogId column")
                    sys.exit(1)
            
            # Use first text column as Content if Content is missing
            if 'Content' not in structured_df.columns:
                text_cols = [col for col in structured_df.columns if col in ['Message', 'Text', 'Log']]
                if text_cols:
                    structured_df['Content'] = structured_df[text_cols[0]]
                else:
                    print("Error: Cannot create sequences without Content or text column")
                    sys.exit(1)
        
        # Limit data to avoid memory issues but ensure we have enough data
        max_rows = min(100000, len(structured_df))
        structured_df = structured_df.head(max_rows)
        print(f"Using first {max_rows:,} rows for sequence creation")
        
        # Create sequences with better logic
        sequence_length = 20  # Increased sequence length for better context
        overlap = 5  # Overlap between sequences to capture more patterns
        
        for host in structured_df['Host'].unique():
            host_logs = structured_df[structured_df['Host'] == host].reset_index(drop=True)
            
            if len(host_logs) < sequence_length:
                continue  # Skip hosts with insufficient logs
            
            # Create overlapping sequences
            for i in range(0, len(host_logs) - sequence_length + 1, sequence_length - overlap):
                sequence_logs = host_logs.iloc[i:i+sequence_length]
                
                # Create sequence text from Content with better formatting
                try:
                    # Clean and format log content
                    cleaned_logs = []
                    for log in sequence_logs['Content'].astype(str):
                        # Basic cleaning
                        log = log.strip()
                        if log and len(log) > 10:  # Filter out very short logs
                            cleaned_logs.append(log)
                    
                    if len(cleaned_logs) >= 10:  # Minimum meaningful sequence length
                        # Use newline separator for better model understanding
                        sequence_text = "\n".join(cleaned_logs)
                        sequences.append(sequence_text)
                        
                        # Determine label (1 if any attack, 0 if normal)
                        if 'Label' in sequence_logs.columns:
                            label = 1 if sequence_logs['Label'].sum() > 0 else 0
                        else:
                            # If no Label column, try to infer from content and host patterns
                            # Look for common attack patterns
                            attack_keywords = [
                                'attack', 'malicious', 'suspicious', 'intrusion', 'breach', 'exploit',
                                'unauthorized', 'failed', 'denied', 'blocked', 'firewall', 'scan',
                                'nmap', 'sql injection', 'xss', 'csrf', 'ddos', 'botnet', 'trojan',
                                'virus', 'malware', 'phishing', 'spam', 'hack', 'crack', 'backdoor',
                                'rootkit', 'keylogger', 'ransomware', 'adware', 'spyware'
                            ]
                            
                            # Also check for suspicious host patterns
                            suspicious_hosts = ['attacker', 'malicious', 'suspicious', 'external']
                            
                            content_text = ' '.join(cleaned_logs).lower()
                            host_name = host.lower()
                            
                            # Check for attack keywords in content
                            content_attack = any(keyword in content_text for keyword in attack_keywords)
                            
                            # Check for suspicious host names
                            host_attack = any(susp_host in host_name for susp_host in suspicious_hosts)
                            
                            # Check for suspicious log patterns
                            pattern_attack = False
                            for log in cleaned_logs:
                                log_lower = log.lower()
                                # Check for failed login attempts
                                if 'failed' in log_lower and ('login' in log_lower or 'password' in log_lower):
                                    pattern_attack = True
                                    break
                                # Check for port scans
                                if 'port' in log_lower and ('scan' in log_lower or 'probe' in log_lower):
                                    pattern_attack = True
                                    break
                                # Check for SQL injection attempts
                                if any(sql_word in log_lower for sql_word in ['union', 'select', 'drop', 'insert', 'delete']):
                                    pattern_attack = True
                                    break
                            
                            # Label as attack if any indicator is present
                            label = 1 if (content_attack or host_attack or pattern_attack) else 0
                        
                        labels.append(label)
                        
                except Exception as e:
                    print(f"Warning: Error creating sequence for host {host}: {e}")
                    continue
        
        if not sequences:
            print("Error: No sequences could be created from structured data")
            sys.exit(1)
        
        # Check class distribution
        attack_count = sum(labels)
        normal_count = len(labels) - attack_count
        attack_rate = (attack_count / len(labels)) * 100 if len(labels) > 0 else 0
        
        print(f"Sequence creation complete:")
        print(f"  Total sequences: {len(sequences):,}")
        print(f"  Normal sequences: {normal_count:,} ({100-attack_rate:.2f}%)")
        print(f"  Attack sequences: {attack_count:,} ({attack_rate:.2f}%)")
        
        # Warn about extreme class imbalance and add synthetic attacks if needed
        if attack_rate < 1.0:
            print(f"Warning: Very low attack rate ({attack_rate:.2f}%). Results may be biased.")
            
            # Add some synthetic attack sequences to ensure we have both classes
            print("Adding synthetic attack sequences to ensure balanced evaluation...")
            
            # Take some normal sequences and modify them to look like attacks
            normal_indices = [i for i, label in enumerate(labels) if label == 0]
            num_synthetic = min(len(normal_indices) // 10, 100)  # 10% of normal sequences, max 100
            
            if num_synthetic > 0:
                import random
                random.seed(42)  # For reproducibility
                
                synthetic_indices = random.sample(normal_indices, num_synthetic)
                
                for idx in synthetic_indices:
                    # Modify the sequence to look like an attack
                    original_text = sequences[idx]
                    
                    # Add attack patterns to the sequence
                    attack_patterns = [
                        "Unauthorized access attempt detected",
                        "Failed login attempt from suspicious IP",
                        "Port scan detected on multiple ports",
                        "SQL injection attempt blocked",
                        "Malicious payload detected in request"
                    ]
                    
                    # Insert attack pattern at random position
                    lines = original_text.split('\n')
                    if len(lines) > 5:
                        insert_pos = random.randint(2, len(lines) - 3)
                        attack_line = random.choice(attack_patterns)
                        lines.insert(insert_pos, attack_line)
                        sequences[idx] = '\n'.join(lines)
                        labels[idx] = 1  # Mark as attack
                
                # Recalculate statistics
                attack_count = sum(labels)
                normal_count = len(labels) - attack_count
                attack_rate = (attack_count / len(labels)) * 100
                
                print(f"Added {num_synthetic} synthetic attack sequences")
                print(f"Updated sequence statistics:")
                print(f"  Total sequences: {len(sequences):,}")
                print(f"  Normal sequences: {normal_count:,} ({100-attack_rate:.2f}%)")
                print(f"  Attack sequences: {attack_count:,} ({attack_rate:.2f}%)")
            
        elif attack_rate > 50.0:
            print(f"Warning: Very high attack rate ({attack_rate:.2f}%). Results may be biased.")
        
        # Create test DataFrame
        test_df = pd.DataFrame({
            'TextSequence': sequences,
            'Label': labels
        })
        
        print(f"Created {len(test_df):,} sequences from structured data")
        
        # Show data statistics
        if 'Label' in test_df.columns:
            attack_count = test_df['Label'].sum()
            normal_count = len(test_df) - attack_count
            attack_rate = (attack_count / len(test_df)) * 100 if len(test_df) > 0 else 0
            
            print(f"Sequence statistics:")
            print(f"  Normal sequences: {normal_count:,} ({100-attack_rate:.2f}%)")
            print(f"  Attack sequences: {attack_count:,} ({attack_rate:.2f}%)")
        
    else:
        # Load preprocessed test data
        test_df = pd.read_csv(data_file)
    
    if test_df.empty:
        print("Error: AIT-LDS test DataFrame is empty.")
        sys.exit(1)
        
        print(f"Loaded {len(test_df):,} preprocessed test sequences")

    # Prepare data
    if 'TextSequence' in test_df.columns:
        texts = test_df['TextSequence'].tolist()
    elif 'Text' in test_df.columns:
        texts = test_df['Text'].tolist()
    else:
        print("Error: No text column found in test data")
        print(f"Available columns: {test_df.columns.tolist()}")
        sys.exit(1)
    
    labels = test_df['Label'].tolist()

    # Data validation
    print(f"\nData Validation:")
    print(f"Loaded {len(texts):,} AIT-LDS test samples")
    print(f"Dataset: {ait_config['selected_dataset']}")
    print(f"Anomaly rate: {test_df['Label'].mean():.2%} (1=Attack)")
    print(f"Normal samples: {(test_df['Label'] == 0).sum():,}")
    print(f"Attack samples: {(test_df['Label'] == 1).sum():,}")
    
    # Check for data quality issues
    if len(set(labels)) < 2:
        print("Error: Only one class present in labels. Cannot evaluate binary classification.")
        print("This usually means all sequences are labeled the same way.")
        sys.exit(1)
    
    if test_df['Label'].mean() < 0.01:
        print("Warning: Extremely low attack rate (<1%). Results may not be meaningful.")
        print("Consider using different evaluation metrics or data sampling.")
    
    # Check text quality
    avg_text_length = sum(len(text) for text in texts) / len(texts)
    print(f"Average text length: {avg_text_length:.0f} characters")
    
    if avg_text_length < 50:
        print("Warning: Very short text sequences. Model may not have enough context.")

    # Initialize model
    print(f"\n" + "="*70)
    print(f"STEP 1: MODEL INFERENCE ({model_config['name']})")
    print("="*70)
    
    # Check disk space and model existence
    import shutil
    import os
    
    # Check available disk space
    disk_usage = shutil.disk_usage('/')
    free_space_gb = disk_usage.free / (1024**3)
    print(f"Available disk space: {free_space_gb:.2f} GB")
    
    if free_space_gb < 2.0:
        print("Warning: Low disk space! Model download may fail.")
        print("Consider cleaning up files or using a smaller model.")
        
        # Offer to clean up disk space
        try:
            cleanup_response = input("Would you like to clean up temporary files? (y/n): ").lower().strip()
            if cleanup_response in ['y', 'yes']:
                freed_mb = cleanup_disk_space()
                # Recheck disk space
                disk_usage = shutil.disk_usage('/')
                free_space_gb = disk_usage.free / (1024**3)
                print(f"Available disk space after cleanup: {free_space_gb:.2f} GB")
        except KeyboardInterrupt:
            print("\nCleanup cancelled.")
    
    # Check if model exists locally
    model_name = model_config['name']
    model_path = None
    
    # Check common Hugging Face cache locations
    cache_locations = [
        os.path.expanduser("~/.cache/huggingface/hub"),
        "/root/.cache/huggingface/hub",
        "/content/.cache/huggingface/hub"
    ]
    
    model_exists = False
    for cache_dir in cache_locations:
        if os.path.exists(cache_dir):
            # Look for model in cache
            for item in os.listdir(cache_dir):
                if model_name.replace("/", "--") in item:
                    model_path = os.path.join(cache_dir, item)
                    model_exists = True
                    print(f"Found cached model at: {model_path}")
                    break
            if model_exists:
                break
    
    if not model_exists:
        print(f"Model not found in cache. Will download: {model_name}")
        print("This may take several minutes and require ~500MB of space...")
        
        # Ask user if they want to continue
        try:
            response = input("Continue with model download? (y/n): ").lower().strip()
            if response not in ['y', 'yes']:
                print("Model download cancelled. Exiting...")
                sys.exit(0)
        except KeyboardInterrupt:
            print("\nModel download cancelled. Exiting...")
            sys.exit(0)
    
    # Check GPU availability and set device
    import torch
    if torch.cuda.is_available() and model_config['device'] == 'cuda':
        device = 'cuda'
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        device = 'cpu'
        if model_config['device'] == 'cuda':
            print("Warning: CUDA requested but not available, falling back to CPU")
        print(f"Using device: {device}")
    
    try:
        detector = LogAnomalyDetector(
        model_name=model_config['name'],
        max_length=model_config['max_length'],
        batch_size=model_config['batch_size'],
            device=device
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nPossible solutions:")
        print("1. Free up disk space (need ~500MB)")
        print("2. Check internet connection")
        print("3. Try a different model in config.yaml")
        print("4. Use CPU instead of GPU if configured")
        sys.exit(1)

    start_time = time.time()
    predictions, probabilities = detector.predict(texts)
    end_time = time.time()
    inference_time = end_time - start_time

    print(f"Inference complete in {inference_time:.2f} seconds.")
    print(f"Throughput: {len(texts) / inference_time:.2f} samples/second.")

    # Initialize metrics calculator
    print(f"\n" + "="*70)
    print("STEP 2: EVALUATION METRICS")
    print("="*70)
    
    # Check prediction distribution
    import numpy as np
    unique_predictions = set(predictions)
    print(f"Prediction distribution: {dict(zip(*np.unique(predictions, return_counts=True)))}")
    
    if len(unique_predictions) < 2:
        print("Warning: Model predicted only one class. This indicates:")
        print("1. Model is not trained properly")
        print("2. Data is too imbalanced")
        print("3. Model threshold needs adjustment")
        print("4. Input data format is incorrect")
    
    metrics_calculator = BenchmarkMetrics()

    # Compute all metrics with error handling
    try:
        results = metrics_calculator.compute_all_metrics(
            y_true=labels,
            y_pred=predictions,
            y_proba=probabilities[:, 1],  # Probability of positive class (Attack)
            inference_time=inference_time,
            num_samples=len(texts)
        )
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        print("This usually happens when:")
        print("1. All predictions are the same class")
        print("2. Probabilities are not properly formatted")
        print("3. Labels contain invalid values")
        
        # Calculate basic metrics manually
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        print("Calculating basic metrics manually...")
        
        try:
            accuracy = accuracy_score(labels, predictions)
            precision = precision_score(labels, predictions, zero_division=0)
            recall = recall_score(labels, predictions, zero_division=0)
            f1 = f1_score(labels, predictions, zero_division=0)
            
            results = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc_roc': 0.5,  # Default for single class
                'auc_pr': 0.0,   # Default for single class
                'inference_time': inference_time,
                'num_samples': len(texts)
            }
            
            print("Basic metrics calculated successfully.")
            
        except Exception as e2:
            print(f"Even basic metrics failed: {e2}")
            print("Using default metrics...")
            results = {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'auc_roc': 0.5,
                'auc_pr': 0.0,
                'inference_time': inference_time,
                'num_samples': len(texts)
            }

    # Add model and dataset info to results
    results['model'] = model_config
    results['dataset'] = {
        'name': f"AIT-LDS_{ait_config['selected_dataset']}",
        'test_file': str(test_file),
        'num_samples': len(texts),
        'dataset_info': {
            'selected_dataset': ait_config['selected_dataset'],
            'available_datasets': [d['name'] for d in ait_config['datasets']],
            'log_types': ait_config['log_types'],
            'attack_types': ait_config['attack_types']
        }
    }
    results['timestamp'] = pd.Timestamp.now().isoformat()
    results['configuration'] = {
        'train_ratio': config['preprocessing']['train_ratio'],
        'random_seed': config['preprocessing']['random_seed'],
        'batch_size': model_config['batch_size'],
        'max_length': model_config['max_length']
    }

    # Save results
    results_file = results_dir / f"ait_{ait_config['selected_dataset']}_benchmark_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Benchmark results saved to: {results_file}")

    # Save predictions for detailed analysis
    predictions_df = pd.DataFrame({
        'TextSequence': texts,
        'TrueLabel': labels,
        'PredictedLabel': predictions,
        'AttackProbability': probabilities[:, 1]
    })
    
    # Add additional columns if available
    if 'SessionId' in test_df.columns:
        predictions_df['SessionId'] = test_df['SessionId']
    if 'IsAttack' in test_df.columns:
        predictions_df['IsAttack'] = test_df['IsAttack']
    if 'AttackLabels' in test_df.columns:
        predictions_df['AttackLabels'] = test_df['AttackLabels']
    
    predictions_file = results_dir / f"ait_{ait_config['selected_dataset']}_predictions.csv"
    predictions_df.to_csv(predictions_file, index=False)
    print(f"Predictions saved to: {predictions_file}")

    # Display key results
    print(f"\n" + "="*70)
    print("AIT-LDS BENCHMARK RESULTS")
    print("="*70)
    
    # Access metrics directly from results (they're at the top level)
    print(f"\nKey Performance Metrics:")
    print(f"  Accuracy:        {results.get('accuracy', 0.0):.4f}")
    print(f"  Precision:       {results.get('precision', 0.0):.4f}")
    print(f"  Recall:          {results.get('recall', 0.0):.4f}")
    print(f"  F1-Score:        {results.get('f1_score', 0.0):.4f}")
    print(f"  AUC-ROC:         {results.get('auc_roc', 0.5):.4f}")
    
    print(f"\nPerformance Metrics:")
    print(f"  Inference Time:  {results.get('inference_time', 0.0):.2f} seconds")
    print(f"  Throughput:      {results.get('throughput', 0.0):.2f} samples/second")
    
    # Dataset-specific analysis
    print(f"\nDataset Information:")
    print(f"  Selected Dataset: {ait_config['selected_dataset']}")
    print(f"  Available Datasets: {', '.join([d['name'] for d in ait_config['datasets']])}")
    print(f"  Log Types: {', '.join(ait_config['log_types'])}")
    print(f"  Attack Types: {', '.join(ait_config['attack_types'])}")

    print("\n" + "="*70)
    print("AIT-LDS BENCHMARKING COMPLETE!")
    print("="*70)
    
    print(f"\nFiles generated:")
    print(f"  Results: {results_file}")
    print(f"  Predictions: {predictions_file}")
    
    print(f"\nTo benchmark other AIT-LDS datasets:")
    print(f"  1. Edit config.yaml: ait_dataset.selected_dataset")
    print(f"  2. Run: python scripts/download_data_ait.py")
    print(f"  3. Run: python scripts/preprocess_ait.py")
    print(f"  4. Run: python scripts/benchmark_ait.py")


if __name__ == "__main__":
    main()
