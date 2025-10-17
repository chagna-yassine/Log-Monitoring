"""
BGL Block-wise Sequence Builder

Groups BGL log events by session/sequence and creates sequences.
BGL doesn't have block IDs like HDFS, so we use session-based grouping.
"""

import re
import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict


class BGLSequenceBuilder:
    """
    Builds session-wise sequences from structured BGL logs.
    
    BGL logs don't have block IDs, so we group by:
    - NodeID + time windows
    - Or by NodeID + session patterns
    """
    
    def __init__(self, session_window_minutes: int = 30):
        """
        Initialize BGL sequence builder.
        
        Args:
            session_window_minutes: Time window for session grouping
        """
        self.session_window_minutes = session_window_minutes
        self.sequences = {}
        
    def extract_session_id(self, row: pd.Series) -> str:
        """
        Extract session ID from BGL log entry.
        
        Args:
            row: Log entry with NodeID, Timestamp, etc.
            
        Returns:
            Session ID string
        """
        node_id = str(row['NodeID'])
        timestamp = str(row['Timestamp'])
        
        # For BGL, create sessions based on NodeID and time windows
        # Extract date and hour from timestamp
        try:
            # Parse timestamp to extract date and hour
            if 'T' in timestamp:
                date_part = timestamp.split('T')[0]
                time_part = timestamp.split('T')[1]
                hour = time_part.split(':')[0]
                
                # Create session based on NodeID + date + hour
                session_id = f"{node_id}_{date_part}_{hour}"
            else:
                # Fallback: use NodeID only
                session_id = f"{node_id}_session"
        except:
            # Fallback: use NodeID only
            session_id = f"{node_id}_session"
        
        return session_id
    
    def build_sequences(
        self,
        structured_csv: str,
        event_id_mapping: Dict[int, int],
        output_csv: str
    ) -> pd.DataFrame:
        """
        Build session-wise sequences from structured BGL log.
        
        Args:
            structured_csv: Path to structured log CSV
            event_id_mapping: Mapping from EventId to template number
            output_csv: Path to save sequences CSV
            
        Returns:
            DataFrame with session sequences
        """
        print(f"\nBuilding session-wise sequences from BGL: {structured_csv}")
        
        # Load structured log
        df = pd.read_csv(structured_csv)
        
        print(f"Loaded {len(df):,} BGL log entries")
        
        # Group events by session ID
        session_to_events = defaultdict(list)
        logs_with_sessions = 0
        logs_without_sessions = 0
        
        for idx, row in df.iterrows():
            session_id = self.extract_session_id(row)
            event_id = int(row['EventId'])
            
            # Map to template number
            template_number = event_id_mapping.get(event_id, 0)
            session_to_events[session_id].append(template_number)
            logs_with_sessions += 1
            
            # Progress update
            if (idx + 1) % 100000 == 0:
                print(f"  Processed {idx + 1:,} logs, found {len(session_to_events):,} sessions")
        
        print(f"\nBGL sequence building complete:")
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
        print(f"\nBGL sequence statistics:")
        print(f"  Total sequences: {len(sequences_df):,}")
        print(f"  Average sequence length: {sequences_df['SequenceLength'].mean():.2f}")
        print(f"  Min sequence length: {sequences_df['SequenceLength'].min()}")
        print(f"  Max sequence length: {sequences_df['SequenceLength'].max()}")
        
        # Save to CSV
        print(f"\nSaving BGL sequences to: {output_csv}")
        sequences_df.to_csv(output_csv, index=False)
        
        return sequences_df
    
    def add_labels(
        self,
        sequences_df: pd.DataFrame,
        labels_csv: str
    ) -> pd.DataFrame:
        """
        Add anomaly labels to BGL sequences.
        
        Args:
            sequences_df: DataFrame with session sequences
            labels_csv: Path to BGL structured CSV with labels
            
        Returns:
            DataFrame with labels added
        """
        print(f"\nAdding BGL labels from: {labels_csv}")
        
        # Load labels from structured CSV
        labels_df = pd.read_csv(labels_csv)
        print(f"Loaded {len(labels_df):,} BGL entries with labels")
        
        # BGL labels are typically in the same structured file
        # We need to group by session and determine if session is anomalous
        if 'Label' in labels_df.columns:
            # Labels are already in the structured file
            print("  Labels found in structured file")
            
            # Create session-based labels
            session_labels = {}
            
            for idx, row in labels_df.iterrows():
                session_id = self.extract_session_id(row)
                label = row['Label']
                
                # If session has any anomaly, mark entire session as anomalous
                if session_id not in session_labels:
                    session_labels[session_id] = label
                else:
                    # If current label is anomaly (1), keep it
                    if label == 1 or str(label).lower() in ['anomaly', 'true']:
                        session_labels[session_id] = 1
            
            # Convert session labels to DataFrame
            session_label_df = pd.DataFrame([
                {'SessionId': session_id, 'Label': label}
                for session_id, label in session_labels.items()
            ])
            
        else:
            print("  Warning: No Label column found in BGL structured file")
            # Create default labels (all normal)
            session_label_df = pd.DataFrame([
                {'SessionId': session_id, 'Label': 0}
                for session_id in sequences_df['SessionId'].unique()
            ])
        
        # Merge labels with sequences
        labeled_df = sequences_df.merge(
            session_label_df,
            on='SessionId',
            how='left'
        )
        
        # Check for missing labels
        missing_labels = labeled_df['Label'].isna().sum()
        if missing_labels > 0:
            print(f"  Warning: {missing_labels} sequences without labels (will be dropped)")
            labeled_df = labeled_df.dropna(subset=['Label'])
        
        # Convert labels to int (handle both string and numeric labels)
        if labeled_df['Label'].dtype == 'object':
            # Map string labels to integers
            label_mapping = {
                'Normal': 0,
                'Anomaly': 1,
                'normal': 0,
                'anomaly': 1,
                0: 0,
                1: 1,
                '0': 0,
                '1': 1,
                'False': 0,
                'True': 1,
                False: 0,
                True: 1
            }
            labeled_df['Label'] = labeled_df['Label'].map(label_mapping)
            
            # Check for unmapped labels
            unmapped = labeled_df['Label'].isna().sum()
            if unmapped > 0:
                print(f"  Warning: {unmapped} labels could not be mapped (will be dropped)")
                labeled_df = labeled_df.dropna(subset=['Label'])
        
        # Ensure labels are integers
        labeled_df['Label'] = labeled_df['Label'].astype(int)
        
        # Statistics
        label_counts = labeled_df['Label'].value_counts().sort_index()
        print(f"\nBGL label distribution:")
        for label, count in label_counts.items():
            label_name = "Normal" if label == 0 else "Anomaly"
            percentage = (count / len(labeled_df)) * 100
            print(f"  {label_name} ({label}): {count:,} ({percentage:.2f}%)")
        
        return labeled_df
    
    def parse_sequence(self, sequence_str: str) -> List[int]:
        """
        Parse sequence string to list of integers.
        
        Args:
            sequence_str: Space-separated sequence string
            
        Returns:
            List of event numbers
        """
        return [int(x) for x in sequence_str.split()]
