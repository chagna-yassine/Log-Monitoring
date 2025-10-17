"""
AIT-LDS Sequence Builder

Builds sequences from AIT-LDS logs using attack-based and time-based grouping.
Groups logs by attack sessions, host sessions, and time windows.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
import json
from datetime import datetime, timedelta
from collections import defaultdict


class AITSequenceBuilder:
    """
    Builds sequences of event IDs for AIT-LDS logs.
    
    AIT-LDS sequences can be built using:
    1. Attack-based grouping (logs from same attack session)
    2. Host-based grouping (logs from same host in time window)
    3. Time-based grouping (logs within time windows)
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ait_config = config['ait_dataset']
        self.output_config = config['output']['ait']
        
        # Setup paths
        self.output_path = Path(self.output_config['base_path'])
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Configuration
        self.time_window_minutes = 30  # Time window for session grouping
        self.sequences = {}
    
    def extract_session_id(self, row: pd.Series, grouping_method: str = "host_time") -> str:
        """
        Extract session ID from AIT log entry.
        
        Args:
            row: Log entry with Host, Timestamp, AttackLabels, etc.
            grouping_method: Method for grouping ("attack", "host_time", "time_only")
            
        Returns:
            Session ID string
        """
        if grouping_method == "attack":
            # Group by attack session
            attack_labels = row.get('AttackLabels', [])
            if pd.notna(attack_labels) and attack_labels:
                # Use attack labels to group
                attack_str = '_'.join(sorted(attack_labels)) if isinstance(attack_labels, list) else str(attack_labels)
                host = row.get('Host', 'unknown')
                return f"attack_{host}_{attack_str}"
            else:
                # No attack - group by host and time
                return self.extract_session_id(row, "host_time")
        
        elif grouping_method == "host_time":
            # Group by host and time window
            host = row.get('Host', 'unknown')
            timestamp = row.get('Timestamp')
            
            if pd.notna(timestamp):
                try:
                    # Parse timestamp and create time window
                    if isinstance(timestamp, str):
                        # Try different timestamp formats
                        for fmt in ['%Y-%m-%dT%H:%M:%S.%fZ', '%Y-%m-%d %H:%M:%S', '%b %d %H:%M:%S']:
                            try:
                                dt = datetime.strptime(timestamp, fmt)
                                break
                            except ValueError:
                                continue
                        else:
                            dt = datetime.now()  # Fallback
                    else:
                        dt = timestamp
                    
                    # Create time window (e.g., 2022-01-24_03)
                    time_window = f"{dt.strftime('%Y-%m-%d_%H')}"
                    return f"host_{host}_{time_window}"
                
                except:
                    # Fallback to host only
                    return f"host_{host}_unknown"
            else:
                return f"host_{host}_no_time"
        
        elif grouping_method == "time_only":
            # Group by time window only
            timestamp = row.get('Timestamp')
            
            if pd.notna(timestamp):
                try:
                    if isinstance(timestamp, str):
                        for fmt in ['%Y-%m-%dT%H:%M:%S.%fZ', '%Y-%m-%d %H:%M:%S', '%b %d %H:%M:%S']:
                            try:
                                dt = datetime.strptime(timestamp, fmt)
                                break
                            except ValueError:
                                continue
                        else:
                            dt = datetime.now()
                    else:
                        dt = timestamp
                    
                    time_window = f"{dt.strftime('%Y-%m-%d_%H')}"
                    return f"time_{time_window}"
                
                except:
                    return "time_unknown"
            else:
                return "time_no_timestamp"
        
        else:
            # Default to host_time
            return self.extract_session_id(row, "host_time")
    
    def build_sequences(
        self,
        structured_csv: str,
        event_id_mapping: Dict[int, int],
        output_csv: str,
        grouping_method: str = "host_time"
    ) -> pd.DataFrame:
        """
        Build sequences from structured AIT logs.
        
        Args:
            structured_csv: Path to structured log CSV
            event_id_mapping: Mapping from EventId to template number
            output_csv: Path to save sequences CSV
            grouping_method: Method for grouping logs into sequences
            
        Returns:
            DataFrame with sequences
        """
        print(f"Building AIT-LDS sequences using {grouping_method} grouping...")
        
        # Load structured log
        df = pd.read_csv(structured_csv)
        
        print(f"Loaded {len(df):,} AIT-LDS log entries")
        
        # Group events by session ID
        session_to_events = defaultdict(list)
        logs_with_sessions = 0
        logs_without_sessions = 0
        
        for idx, row in df.iterrows():
            session_id = self.extract_session_id(row, grouping_method)
            event_id = row.get('EventId')
            
            if pd.notna(event_id):
                # Map to template number
                template_number = event_id_mapping.get(int(event_id), 0)
                session_to_events[session_id].append(template_number)
                logs_with_sessions += 1
            else:
                logs_without_sessions += 1
            
            # Progress update
            if (idx + 1) % 100000 == 0:
                print(f"  Processed {idx + 1:,} logs, found {len(session_to_events):,} sessions")
        
        print(f"\nAIT-LDS sequence building complete:")
        print(f"  Logs with sessions: {logs_with_sessions:,}")
        print(f"  Logs without sessions: {logs_without_sessions:,}")
        print(f"  Unique sessions: {len(session_to_events):,}")
        
        # Create sequences DataFrame
        sequences = []
        for session_id, event_numbers in session_to_events.items():
            sequences.append({
                'SessionId': session_id,
                'EventSequence': ' '.join(map(str, event_numbers)),
                'SequenceLength': len(event_numbers)
            })
        
        sequences_df = pd.DataFrame(sequences)
        
        # Sort by session ID
        sequences_df = sequences_df.sort_values('SessionId').reset_index(drop=True)
        
        # Statistics
        print(f"\nAIT-LDS sequence statistics:")
        print(f"  Total sequences: {len(sequences_df):,}")
        print(f"  Average sequence length: {sequences_df['SequenceLength'].mean():.2f}")
        print(f"  Min sequence length: {sequences_df['SequenceLength'].min()}")
        print(f"  Max sequence length: {sequences_df['SequenceLength'].max()}")
        
        # Save sequences
        print(f"\nSaving AIT-LDS sequences to: {output_csv}")
        sequences_df.to_csv(output_csv, index=False)
        
        return sequences_df
    
    def add_labels(
        self,
        sequences_df: pd.DataFrame,
        structured_csv: str
    ) -> pd.DataFrame:
        """
        Add anomaly labels to AIT-LDS sequences.
        
        Args:
            sequences_df: DataFrame with session sequences
            structured_csv: Path to structured log CSV with labels
            
        Returns:
            DataFrame with labels added
        """
        print(f"\nAdding AIT-LDS labels from: {structured_csv}")
        
        # Load structured logs with labels
        df = pd.read_csv(structured_csv)
        print(f"Loaded {len(df):,} structured AIT-LDS entries")
        
        # Check if Label column exists
        if 'Label' not in df.columns:
            print("  Warning: No Label column found in structured file")
            print("  Creating default labels (all normal)")
            
            labeled_df = sequences_df.copy()
            labeled_df['Label'] = 0
            
            print(f"  Created default labels for {len(labeled_df):,} sequences")
            return labeled_df
        
        # Create session-based labels from structured data
        session_labels = {}
        
        for idx, row in df.iterrows():
            session_id = self.extract_session_id(row)
            label = row['Label']
            attack_labels = row.get('AttackLabels', [])
            
            # If session has any attack, mark entire session as anomalous
            if session_id not in session_labels:
                session_labels[session_id] = {
                    'label': label,
                    'attack_labels': attack_labels,
                    'is_attack': bool(label == 1 or (pd.notna(attack_labels) and attack_labels))
                }
            else:
                # If current entry is attack, mark session as attack
                is_current_attack = bool(label == 1 or (pd.notna(attack_labels) and attack_labels))
                if is_current_attack:
                    session_labels[session_id]['label'] = 1
                    session_labels[session_id]['is_attack'] = True
                    if pd.notna(attack_labels) and attack_labels:
                        session_labels[session_id]['attack_labels'] = attack_labels
            
            # Progress update
            if (idx + 1) % 100000 == 0:
                print(f"  Processed {idx + 1:,} structured entries")
        
        # Convert session labels to DataFrame
        session_label_df = pd.DataFrame([
            {
                'SessionId': session_id,
                'Label': data['label'],
                'IsAttack': data['is_attack'],
                'AttackLabels': data['attack_labels']
            }
            for session_id, data in session_labels.items()
        ])
        
        print(f"Created labels for {len(session_label_df):,} sessions")
        
        # Merge labels with sequences
        labeled_df = sequences_df.merge(session_label_df, on='SessionId', how='left')
        
        # Handle missing labels
        missing_labels = labeled_df['Label'].isna().sum()
        if missing_labels > 0:
            print(f"  Warning: {missing_labels} sequences without labels (will be dropped)")
            labeled_df = labeled_df.dropna(subset=['Label'])
        
        # Convert labels to int
        labeled_df['Label'] = labeled_df['Label'].astype(int)
        labeled_df['IsAttack'] = labeled_df['IsAttack'].fillna(False).astype(bool)
        
        # Statistics
        label_counts = labeled_df['Label'].value_counts().sort_index()
        print(f"\nAIT-LDS label distribution:")
        for label, count in label_counts.items():
            label_name = "Normal" if label == 0 else "Attack"
            percentage = (count / len(labeled_df)) * 100
            print(f"  {label_name} ({label}): {count:,} ({percentage:.2f}%)")
        
        # Attack type statistics
        if 'AttackLabels' in labeled_df.columns:
            attack_sessions = labeled_df[labeled_df['IsAttack'] == True]
            if len(attack_sessions) > 0:
                print(f"\nAttack session details:")
                print(f"  Total attack sessions: {len(attack_sessions):,}")
                
                # Count attack types
                attack_types = defaultdict(int)
                for attack_labels in attack_sessions['AttackLabels'].dropna():
                    if isinstance(attack_labels, list):
                        for attack_type in attack_labels:
                            attack_types[attack_type] += 1
                
                if attack_types:
                    print(f"  Attack types found:")
                    for attack_type, count in sorted(attack_types.items()):
                        print(f"    {attack_type}: {count}")
        
        return labeled_df
    
    def save_labeled_sequences(self, labeled_df: pd.DataFrame, output_csv: str) -> None:
        """Save labeled sequences to CSV file."""
        print(f"Saving labeled AIT-LDS sequences to: {output_csv}")
        labeled_df.to_csv(output_csv, index=False)
        
        # Save summary statistics
        summary_file = Path(output_csv).with_suffix('.summary.json')
        summary = {
            'total_sequences': len(labeled_df),
            'normal_sequences': len(labeled_df[labeled_df['Label'] == 0]),
            'attack_sequences': len(labeled_df[labeled_df['Label'] == 1]),
            'attack_rate': (labeled_df['Label'].sum() / len(labeled_df)) * 100,
            'avg_sequence_length': labeled_df['SequenceLength'].mean(),
            'min_sequence_length': labeled_df['SequenceLength'].min(),
            'max_sequence_length': labeled_df['SequenceLength'].max(),
            'grouping_method': 'host_time'  # Could be made configurable
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Summary saved to: {summary_file}")
