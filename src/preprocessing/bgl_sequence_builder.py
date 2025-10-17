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
        
    def extract_block_id(self, content: str) -> str:
        """
        Extract block/session ID from BGL log content.
        BGL logs may contain job IDs, node IDs, or other identifiers.
        
        Args:
            content: Log content string
            
        Returns:
            Block/session ID or None if not found
        """
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
        
        # Group events by block ID (similar to HDFS)
        block_to_events = defaultdict(list)
        logs_with_blocks = 0
        logs_without_blocks = 0
        
        for idx, row in df.iterrows():
            content = str(row['Content'])
            block_id = self.extract_block_id(content)
            
            if block_id:
                event_id = int(row['EventId'])
                # Map to template number
                template_number = event_id_mapping.get(event_id, 0)
                block_to_events[block_id].append(template_number)
                logs_with_blocks += 1
            else:
                logs_without_blocks += 1
            
            # Progress update
            if (idx + 1) % 100000 == 0:
                print(f"  Processed {idx + 1:,} logs, found {len(block_to_events):,} blocks")
        
        print(f"\nBGL sequence building complete:")
        print(f"  Logs with block IDs: {logs_with_blocks:,}")
        print(f"  Logs without block IDs: {logs_without_blocks:,}")
        print(f"  Unique blocks: {len(block_to_events):,}")
        
        # Create sequences DataFrame
        sequences = []
        for block_id, event_numbers in block_to_events.items():
            sequences.append({
                'BlockId': block_id,
                'EventSequence': ' '.join(map(str, event_numbers)),
                'SequenceLength': len(event_numbers)
            })
        
        sequences_df = pd.DataFrame(sequences)
        
        # Sort by block ID
        sequences_df = sequences_df.sort_values('BlockId').reset_index(drop=True)
        
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
            sequences_df: DataFrame with block sequences
            labels_csv: Path to BGL anomaly label CSV (similar to HDFS)
            
        Returns:
            DataFrame with labels added
        """
        print(f"\nAdding BGL labels from: {labels_csv}")
        
        # Load labels (should be similar to HDFS format)
        labels_df = pd.read_csv(labels_csv)
        print(f"Loaded {len(labels_df):,} BGL labels")
        
        # Rename columns if needed (similar to HDFS)
        if 'BlockId' in labels_df.columns and 'Label' in labels_df.columns:
            pass
        elif len(labels_df.columns) == 2:
            labels_df.columns = ['BlockId', 'Label']
        else:
            raise ValueError(f"Unexpected BGL label file format. Columns: {labels_df.columns.tolist()}")
        
        # Merge labels with sequences
        labeled_df = sequences_df.merge(
            labels_df,
            on='BlockId',
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
