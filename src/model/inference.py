"""
Model Inference

Loads and runs inference with the log anomaly detection model.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Tuple, Dict
import numpy as np
from tqdm import tqdm


class LogAnomalyDetector:
    """
    Wrapper for log anomaly detection model inference.
    """
    
    def __init__(
        self,
        model_name: str = "Dumi2025/log-anomaly-detection-model-roberta",
        max_length: int = 512,
        batch_size: int = 32,
        device: str = None
    ):
        """
        Initialize the log anomaly detector.
        
        Args:
            model_name: Hugging Face model name
            max_length: Maximum sequence length for tokenization
            batch_size: Batch size for inference
            device: Device to run on ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"\nLoading model: {model_name}")
        print(f"Device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully")
        print(f"  Max length: {max_length}")
        print(f"  Batch size: {batch_size}")
        
    def predict(
        self,
        texts: List[str],
        return_probabilities: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run inference on a list of texts.
        
        Args:
            texts: List of text sequences
            return_probabilities: Whether to return probabilities or logits
            
        Returns:
            Tuple of (predictions, probabilities/logits)
        """
        all_predictions = []
        all_probabilities = []
        
        # Process in batches
        num_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        
        with torch.no_grad():
            for i in tqdm(range(num_batches), desc="Running inference"):
                # Get batch
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, len(texts))
                batch_texts = texts[start_idx:end_idx]
                
                # Tokenize
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                # Move to device
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits.cpu().numpy()
                
                # Get predictions
                predictions = np.argmax(logits, axis=1)
                all_predictions.extend(predictions)
                
                # Get probabilities
                if return_probabilities:
                    probabilities = torch.softmax(
                        torch.tensor(logits),
                        dim=1
                    ).numpy()
                    all_probabilities.extend(probabilities)
                else:
                    all_probabilities.extend(logits)
        
        return np.array(all_predictions), np.array(all_probabilities)
    
    def predict_single(self, text: str) -> Tuple[int, np.ndarray]:
        """
        Run inference on a single text.
        
        Args:
            text: Text sequence
            
        Returns:
            Tuple of (prediction, probabilities)
        """
        predictions, probabilities = self.predict([text])
        return predictions[0], probabilities[0]
    
    def get_tokenizer(self):
        """Get the tokenizer instance."""
        return self.tokenizer
    
    def get_model_info(self) -> Dict:
        """
        Get model information.
        
        Returns:
            Dictionary with model details
        """
        return {
            'model_name': self.model_name,
            'device': self.device,
            'max_length': self.max_length,
            'batch_size': self.batch_size,
            'num_labels': self.model.config.num_labels,
            'model_type': self.model.config.model_type
        }

