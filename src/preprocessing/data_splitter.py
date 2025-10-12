"""
Data Splitter

Splits data into train/test sets following LogBERT methodology.
"""

import pandas as pd
import numpy as np
from typing import Tuple
from pathlib import Path


class DataSplitter:
    """
    Splits HDFS sequences into train/test sets.
    
    Following LogBERT methodology:
    - 70% of normal blocks for training
    - 30% of normal blocks + all anomalous blocks for testing
    """
    
    def __init__(self, train_ratio: float = 0.7, random_seed: int = 42):
        """
        Initialize data splitter.
        
        Args:
            train_ratio: Ratio of normal blocks to use for training
            random_seed: Random seed for reproducibility
        """
        self.train_ratio = train_ratio
        self.random_seed = random_seed
        
    def split_data(
        self,
        sequences_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split sequences into train and test sets.
        
        Args:
            sequences_df: DataFrame with BlockId, EventSequence, SequenceLength, Label
            
        Returns:
            Tuple of (train_df, test_df)
        """
        print(f"\nSplitting data (train_ratio={self.train_ratio}, seed={self.random_seed})")
        
        # Separate normal and anomalous blocks
        normal_df = sequences_df[sequences_df['Label'] == 0].copy()
        anomaly_df = sequences_df[sequences_df['Label'] == 1].copy()
        
        print(f"  Normal blocks: {len(normal_df):,}")
        print(f"  Anomalous blocks: {len(anomaly_df):,}")
        
        # Shuffle normal blocks
        np.random.seed(self.random_seed)
        normal_df = normal_df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        
        # Split normal blocks
        train_size = int(len(normal_df) * self.train_ratio)
        train_normal = normal_df[:train_size]
        test_normal = normal_df[train_size:]
        
        # Test set = remaining normal + all anomalous
        train_df = train_normal
        test_df = pd.concat([test_normal, anomaly_df], ignore_index=True)
        
        # Shuffle test set
        test_df = test_df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        
        print(f"\nSplit complete:")
        print(f"  Train set: {len(train_df):,} (all normal)")
        print(f"  Test set: {len(test_df):,} ({len(test_normal):,} normal, {len(anomaly_df):,} anomalous)")
        
        # Test set statistics
        test_anomaly_rate = (len(anomaly_df) / len(test_df)) * 100
        print(f"  Test anomaly rate: {test_anomaly_rate:.2f}%")
        
        return train_df, test_df
    
    def save_splits(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        output_dir: str
    ) -> None:
        """
        Save train and test splits to CSV files.
        
        Args:
            train_df: Training data
            test_df: Test data
            output_dir: Directory to save files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        train_file = output_path / "hdfs_train.csv"
        test_file = output_path / "hdfs_test.csv"
        
        print(f"\nSaving splits:")
        print(f"  Train: {train_file}")
        train_df.to_csv(train_file, index=False)
        
        print(f"  Test: {test_file}")
        test_df.to_csv(test_file, index=False)
        
    def load_test_set(self, test_file: str) -> pd.DataFrame:
        """
        Load test set from CSV.
        
        Args:
            test_file: Path to test CSV file
            
        Returns:
            Test DataFrame
        """
        print(f"Loading test set from: {test_file}")
        test_df = pd.read_csv(test_file)
        print(f"Loaded {len(test_df):,} test samples")
        
        # Statistics
        label_counts = test_df['Label'].value_counts().sort_index()
        for label, count in label_counts.items():
            label_name = "Normal" if label == 0 else "Anomaly"
            percentage = (count / len(test_df)) * 100
            print(f"  {label_name}: {count:,} ({percentage:.2f}%)")
        
        return test_df

