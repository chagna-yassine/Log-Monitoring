"""
Text Converter

Converts numerical event sequences to readable text for model input.
"""

from typing import List
import pandas as pd
from .template_mapper import TemplateMapper


class TextConverter:
    """
    Converts numerical event sequences to text format for LLM input.
    """
    
    def __init__(self, template_mapper: TemplateMapper, separator: str = " | "):
        """
        Initialize text converter.
        
        Args:
            template_mapper: TemplateMapper instance with loaded mapping
            separator: Separator for joining event templates
        """
        self.template_mapper = template_mapper
        self.separator = separator
        
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
        templates = [
            self.template_mapper.map_number_to_template(num)
            for num in numbers
        ]
        
        # Join with separator
        return self.separator.join(templates)
    
    def convert_dataframe(
        self,
        df: pd.DataFrame,
        sequence_column: str = 'EventSequence'
    ) -> pd.DataFrame:
        """
        Convert all sequences in a DataFrame to text.
        
        Args:
            df: DataFrame with sequence column
            sequence_column: Name of column containing sequences
            
        Returns:
            DataFrame with added 'Text' column
        """
        print(f"\nConverting sequences to text...")
        
        df = df.copy()
        df['Text'] = df[sequence_column].apply(self.convert_sequence)
        
        # Statistics
        empty_texts = (df['Text'] == "").sum()
        if empty_texts > 0:
            print(f"  Warning: {empty_texts} sequences converted to empty text")
        
        # Show sample
        print(f"\nSample converted texts:")
        for idx, row in df.head(3).iterrows():
            text = row['Text']
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

