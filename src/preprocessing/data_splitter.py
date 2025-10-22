"""
Data Splitter

Splits data into train/test sets following LogBERT methodology.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from pathlib import Path


class DataSplitter:
    """
    Splits sequences into train/test sets.
    
    Following LogBERT methodology:
    - 70% of normal blocks for training
    - 30% of normal blocks + all anomalous blocks for testing
    """
    
    def __init__(self, config: Dict[str, Any], dataset_type: str = "hdfs"):
        """
        Initialize data splitter.
        
        Args:
            config: Configuration dictionary
            dataset_type: Type of dataset ("hdfs", "bgl", "ait")
        """
        self.config = config
        self.dataset_type = dataset_type
        self.train_ratio = config['preprocessing']['train_ratio']
        self.random_seed = config['preprocessing']['random_seed']
        
    def split_data(
        self,
        sequences_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split sequences into train and test sets.
        
        Args:
            sequences_df: DataFrame with EventSequence, SequenceLength, Label
            
        Returns:
            Tuple of (train_df, test_df)
        """
        print(f"\nSplitting {self.dataset_type.upper()} data (train_ratio={self.train_ratio}, seed={self.random_seed})")
        
        # Separate normal and anomalous sequences
        normal_df = sequences_df[sequences_df['Label'] == 0].copy()
        anomaly_df = sequences_df[sequences_df['Label'] == 1].copy()
        
        print(f"  Normal sequences: {len(normal_df):,}")
        print(f"  Anomalous sequences: {len(anomaly_df):,}")
        
        # Shuffle normal sequences
        np.random.seed(self.random_seed)
        normal_df = normal_df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        
        # Split normal sequences
        train_size = int(len(normal_df) * self.train_ratio)
        train_normal = normal_df[:train_size]
        test_normal = normal_df[train_size:]
        
        # Test set = remaining normal + all anomalous
        test_df = pd.concat([test_normal, anomaly_df], ignore_index=True)
        
        # Shuffle test set
        test_df = test_df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        
        print(f"  Train set: {len(train_normal):,} sequences")
        print(f"  Test set: {len(test_df):,} sequences")
        
        # Save train and test sets
        output_config = self.config['output'][self.dataset_type]
        output_path = Path(output_config['base_path'])
        output_path.mkdir(parents=True, exist_ok=True)
        
        train_file = output_path / f"{self.dataset_type}_train.csv"
        test_file = output_path / f"{self.dataset_type}_test.csv"
        
        train_normal.to_csv(train_file, index=False)
        test_df.to_csv(test_file, index=False)
        
        print(f"  Train set saved to: {train_file}")
        print(f"  Test set saved to: {test_file}")
        
        return train_normal, test_df

