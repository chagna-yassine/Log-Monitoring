#!/usr/bin/env python3
"""
Script to upload processed AIT Fox dataset to Hugging Face Hub.
Memory-efficient implementation to handle large datasets within 12GB RAM limit.
"""

import os
import gc
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Generator
import psutil
import time
from huggingface_hub import HfApi, create_repo
from datasets import Dataset, DatasetDict, Features, Value, ClassLabel
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MemoryEfficientUploader:
    """Memory-efficient dataset uploader for Hugging Face Hub."""
    
    def __init__(self, 
                 dataset_path: str,
                 repo_name: str,
                 chunk_size: int = 10000,
                 max_memory_gb: float = 10.0):
        """
        Initialize the uploader.
        
        Args:
            dataset_path: Path to the processed dataset directory
            repo_name: Hugging Face repository name
            chunk_size: Number of rows to process at once
            max_memory_gb: Maximum memory usage in GB before triggering cleanup
        """
        self.dataset_path = Path(dataset_path)
        self.repo_name = repo_name
        self.chunk_size = chunk_size
        self.max_memory_gb = max_memory_gb
        self.api = HfApi()
        
    def get_memory_usage(self) -> float:
        """Get current memory usage in GB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024**3
    
    def memory_cleanup(self):
        """Force garbage collection to free memory."""
        gc.collect()
        logger.info(f"Memory usage after cleanup: {self.get_memory_usage():.2f} GB")
    
    def check_memory_limit(self):
        """Check if memory usage exceeds limit and cleanup if needed."""
        current_memory = self.get_memory_usage()
        if current_memory > self.max_memory_gb:
            logger.warning(f"Memory usage {current_memory:.2f} GB exceeds limit {self.max_memory_gb} GB")
            self.memory_cleanup()
            
    def load_file_in_chunks(self, file_path: Path, chunk_size: int = None) -> Generator[pd.DataFrame, None, None]:
        """Load CSV file in chunks to manage memory."""
        if chunk_size is None:
            chunk_size = self.chunk_size
            
        logger.info(f"Loading {file_path.name} in chunks of {chunk_size} rows")
        
        try:
            # Use pandas chunking for large files
            for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                yield chunk
                self.check_memory_limit()
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            raise
    
    def get_file_size_mb(self, file_path: Path) -> float:
        """Get file size in MB."""
        return file_path.stat().st_size / (1024 * 1024)
    
    def create_dataset_features(self, sample_data: pd.DataFrame) -> Features:
        """Create Hugging Face dataset features from sample data."""
        features = Features({
            'SessionId': Value('string'),
            'EventSequence': Value('string'),
            'SequenceLength': Value('int64'),
            'Label': Value('int64'),
            'IsAttack': Value('bool'),
            'AttackLabels': Value('string'),
            'TextSequence': Value('string')
        })
        return features
    
    def process_chunk_to_dict(self, chunk: pd.DataFrame) -> Dict[str, List]:
        """Convert pandas chunk to dictionary format for Hugging Face."""
        # Ensure all columns are properly formatted
        chunk_dict = {
            'SessionId': chunk['SessionId'].astype(str).tolist(),
            'EventSequence': chunk['EventSequence'].astype(str).tolist(),
            'SequenceLength': chunk['SequenceLength'].astype('int64').tolist(),
            'Label': chunk['Label'].astype('int64').tolist(),
            'IsAttack': chunk['IsAttack'].astype(bool).tolist(),
            'AttackLabels': chunk['AttackLabels'].fillna('').astype(str).tolist(),
            'TextSequence': chunk['TextSequence'].astype(str).tolist()
        }
        return chunk_dict
    
    def create_dataset_from_chunks(self, file_path: Path) -> Dataset:
        """Create Hugging Face dataset from chunks."""
        logger.info(f"Creating dataset from {file_path.name}")
        
        # Get sample data to determine features
        sample_chunk = next(pd.read_csv(file_path, nrows=100))
        features = self.create_dataset_features(sample_chunk)
        
        # Process file in chunks and accumulate data
        all_data = {
            'SessionId': [],
            'EventSequence': [],
            'SequenceLength': [],
            'Label': [],
            'IsAttack': [],
            'AttackLabels': [],
            'TextSequence': []
        }
        
        total_rows = 0
        for chunk in self.load_file_in_chunks(file_path):
            chunk_dict = self.process_chunk_to_dict(chunk)
            
            # Append chunk data to accumulator
            for key in all_data:
                all_data[key].extend(chunk_dict[key])
            
            total_rows += len(chunk)
            logger.info(f"Processed {total_rows} rows, Memory: {self.get_memory_usage():.2f} GB")
            
            # Memory cleanup every few chunks
            if total_rows % (self.chunk_size * 5) == 0:
                self.memory_cleanup()
        
        logger.info(f"Total rows processed: {total_rows}")
        
        # Create dataset from accumulated data
        dataset = Dataset.from_dict(all_data, features=features)
        return dataset
    
    def create_dataset_dict(self) -> DatasetDict:
        """Create DatasetDict with train and test splits."""
        logger.info("Creating dataset dictionary with train/test splits")
        
        # Create train dataset
        train_file = self.dataset_path / "ait_train.csv"
        test_file = self.dataset_path / "ait_test.csv"
        
        if not train_file.exists() or not test_file.exists():
            raise FileNotFoundError(f"Required files not found: {train_file} or {test_file}")
        
        # Process train dataset
        logger.info(f"Processing train dataset: {train_file.name}")
        train_dataset = self.create_dataset_from_chunks(train_file)
        self.memory_cleanup()
        
        # Process test dataset
        logger.info(f"Processing test dataset: {test_file.name}")
        test_dataset = self.create_dataset_from_chunks(test_file)
        self.memory_cleanup()
        
        # Create dataset dictionary
        dataset_dict = DatasetDict({
            'train': train_dataset,
            'test': test_dataset
        })
        
        return dataset_dict
    
    def upload_dataset(self, private: bool = True) -> str:
        """
        Upload dataset to Hugging Face Hub.
        
        Args:
            private: Whether to create a private repository
            
        Returns:
            Repository URL
        """
        logger.info(f"Starting upload process for repository: {self.repo_name}")
        
        try:
            # Create repository if it doesn't exist
            logger.info(f"Creating repository: {self.repo_name}")
            create_repo(
                repo_id=self.repo_name,
                token=os.getenv('HF_TOKEN'),
                private=private,
                repo_type='dataset'
            )
            
            # Create dataset
            logger.info("Creating dataset from files")
            dataset_dict = self.create_dataset_dict()
            
            # Upload dataset
            logger.info("Uploading dataset to Hugging Face Hub")
            dataset_dict.push_to_hub(
                repo_id=self.repo_name,
                token=os.getenv('HF_TOKEN')
            )
            
            repo_url = f"https://huggingface.co/datasets/{self.repo_name}"
            logger.info(f"Dataset uploaded successfully: {repo_url}")
            
            return repo_url
            
        except Exception as e:
            logger.error(f"Error uploading dataset: {e}")
            raise
    
    def upload_metadata_files(self):
        """Upload additional metadata files."""
        logger.info("Uploading metadata files")
        
        metadata_files = [
            "ait_log_templates.json",
            "ait_sequence_labeled.summary.json",
            "ait_templates.csv",
            "ait_structured.csv"
        ]
        
        for file_name in metadata_files:
            file_path = self.dataset_path / file_name
            if file_path.exists():
                logger.info(f"Uploading {file_name}")
                self.api.upload_file(
                    path_or_fileobj=str(file_path),
                    path_in_repo=f"metadata/{file_name}",
                    repo_id=self.repo_name,
                    token=os.getenv('HF_TOKEN'),
                    repo_type='dataset'
                )
            else:
                logger.warning(f"Metadata file not found: {file_name}")

def main():
    """Main function to upload the dataset."""
    
    # Configuration
    DATASET_PATH = "datasets/ait/output/ait"
    REPO_NAME = "ait-fox-log-anomaly-dataset"  # Change this to your desired repo name
    CHUNK_SIZE = 5000  # Adjust based on your RAM
    MAX_MEMORY_GB = 10.0  # Leave 2GB buffer for system
    
    # Check if HF token is set
    if not os.getenv('HF_TOKEN'):
        logger.error("HF_TOKEN environment variable not set!")
        logger.info("Please set your Hugging Face token:")
        logger.info("export HF_TOKEN=your_token_here")
        return
    
    # Check if dataset directory exists
    dataset_path = Path(DATASET_PATH)
    if not dataset_path.exists():
        logger.error(f"Dataset directory not found: {dataset_path}")
        return
    
    # Display file information
    logger.info("Dataset files:")
    for file_path in dataset_path.glob("*.csv"):
        size_mb = file_path.stat().st_size / (1024 * 1024)
        logger.info(f"  {file_path.name}: {size_mb:.2f} MB")
    
    # Initialize uploader
    uploader = MemoryEfficientUploader(
        dataset_path=DATASET_PATH,
        repo_name=REPO_NAME,
        chunk_size=CHUNK_SIZE,
        max_memory_gb=MAX_MEMORY_GB
    )
    
    try:
        # Upload dataset
        repo_url = uploader.upload_dataset(private=True)
        
        # Upload metadata files
        uploader.upload_metadata_files()
        
        logger.info("="*70)
        logger.info("UPLOAD COMPLETED SUCCESSFULLY!")
        logger.info(f"Repository URL: {repo_url}")
        logger.info("="*70)
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise

if __name__ == "__main__":
    main()
