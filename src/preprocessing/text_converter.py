"""
Text Converter

Converts numerical event sequences to readable text for model input.
"""

from typing import List, Dict, Any
import pandas as pd
import json
from pathlib import Path


class TextConverter:
    """
    Converts numerical event sequences to text format for LLM input.
    """
    
    def __init__(self, config: Dict[str, Any], dataset_type: str = "hdfs"):
        """
        Initialize text converter.
        
        Args:
            config: Configuration dictionary
            dataset_type: Type of dataset ("hdfs", "bgl", "ait")
        """
        self.config = config
        self.dataset_type = dataset_type
        self.separator = config['preprocessing'].get('sequence_separator', ' | ')
        
        # Load template mapping
        self.template_mapping = self._load_template_mapping()
        
    def _load_template_mapping(self) -> Dict[int, str]:
        """Load template mapping from file."""
        output_config = self.config['output'][self.dataset_type]
        output_path = Path(output_config['base_path'])
        template_mapping_file = output_path / output_config['template_mapping']
        
        if template_mapping_file.exists():
            with open(template_mapping_file, 'r') as f:
                mapping = json.load(f)
            
            # Handle nested structure from TemplateMapper
            if 'number_to_template' in mapping:
                return {int(k): v for k, v in mapping['number_to_template'].items()}
            else:
                # Direct mapping format
                return {int(k): v for k, v in mapping.items()}
        else:
            print(f"Warning: Template mapping file not found: {template_mapping_file}")
            return {}
    
    def convert_sequence(self, sequence_str: str) -> str:
        """
        Convert a numerical sequence to text.
        
        Args:
            sequence_str: Space-separated sequence of event numbers
            
        Returns:
            Text sequence with event templates
        """
        # Parse sequence string
        try:
            numbers = [int(x) for x in sequence_str.split()]
        except (ValueError, AttributeError):
            return ""
        
        # Map numbers to templates
        templates = []
        for num in numbers:
            template = self.template_mapping.get(num, f"<UNKNOWN_{num}>")
            templates.append(template)
        
        # Join with separator
        return self.separator.join(templates)
    
    def convert_sequences_to_text(self, sequences_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert all sequences in a DataFrame to text.
        
        Args:
            sequences_df: DataFrame with EventSequence column
            
        Returns:
            DataFrame with added 'TextSequence' column
        """
        print(f"\nConverting {self.dataset_type.upper()} sequences to text...")
        
        df = sequences_df.copy()
        df['TextSequence'] = df['EventSequence'].apply(self.convert_sequence)
        
        # Statistics
        empty_texts = (df['TextSequence'] == "").sum()
        if empty_texts > 0:
            print(f"  Warning: {empty_texts} sequences converted to empty text")
        
        # Show sample
        print(f"\nSample converted texts:")
        for idx, row in df.head(3).iterrows():
            text = row['TextSequence']
            if len(text) > 200:
                text = text[:200] + "..."
            print(f"  [{idx}] {text}")
        
        return df
    
    def get_max_token_length(self, texts: List[str], tokenizer) -> int:
        """
        Get maximum token length in the dataset.
        
        Args:
            texts: List of text sequences
            tokenizer: Tokenizer instance
            
        Returns:
            Maximum token length
        """
        print("\nAnalyzing token lengths...")
        
        sample_size = min(1000, len(texts))
        sample_texts = texts[:sample_size]
        
        token_lengths = []
        for text in sample_texts:
            tokens = tokenizer.encode(text, add_special_tokens=True)
            token_lengths.append(len(tokens))
        
        max_len = max(token_lengths)
        avg_len = sum(token_lengths) / len(token_lengths)
        
        print(f"  Sample size: {sample_size}")
        print(f"  Average token length: {avg_len:.2f}")
        print(f"  Max token length: {max_len}")
        
        return max_len

