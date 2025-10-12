"""
Example Usage

Demonstrates how to use the benchmarking system programmatically.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from preprocessing import (
    HDFSLogParser,
    TemplateMapper,
    SequenceBuilder,
    DataSplitter,
    TextConverter
)
from model import LogAnomalyDetector
from evaluation import BenchmarkMetrics


def example_parse_logs():
    """Example: Parse raw logs with Drain algorithm."""
    print("="*70)
    print("EXAMPLE 1: Parsing Logs")
    print("="*70)
    
    # Initialize parser
    parser = HDFSLogParser(
        depth=4,
        st=0.4,
        rex=[
            r"(?<=blk_)[-\d]+",
            r"\d+\.\d+\.\d+\.\d+",
            r"(/[-\w]+)+"
        ]
    )
    
    # Parse a small sample
    sample_logs = [
        "081109 203615 148 INFO dfs.DataNode$PacketResponder: Received block blk_38",
        "081109 203615 148 INFO dfs.DataNode$DataXceiver: Receiving block blk_38",
        "081109 203616 148 INFO dfs.DataNode$PacketResponder: Received block blk_39"
    ]
    
    for i, log in enumerate(sample_logs):
        parsed = parser.parse_line(log)
        if parsed:
            event_id, template = parser.drain.parse(i, parsed['Content'])
            print(f"\nLog: {log}")
            print(f"  Event ID: {event_id}")
            print(f"  Template: {template}")


def example_template_mapping():
    """Example: Create and use template mapping."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Template Mapping")
    print("="*70)
    
    # Create mapper
    mapper = TemplateMapper()
    
    # Example mapping
    mapper.number_to_template = {
        1: "Receiving block <*>",
        2: "PacketResponder for block <*>",
        3: "Verification succeeded for <*>"
    }
    
    # Convert sequence
    sequence = [1, 2, 3, 1]
    text = mapper.get_template_text(sequence)
    
    print(f"\nSequence: {sequence}")
    print(f"Text: {text}")


def example_model_inference():
    """Example: Run inference with the model."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Model Inference")
    print("="*70)
    
    # Initialize detector
    print("\nInitializing model...")
    detector = LogAnomalyDetector(
        model_name="Dumi2025/log-anomaly-detection-model-roberta",
        max_length=512,
        batch_size=4,
        device="cpu"  # Use CPU for example
    )
    
    # Example texts
    texts = [
        "Receiving block | PacketResponder for block | Verification succeeded",
        "Connection timeout | Failed to read | Error occurred | Connection lost"
    ]
    
    print("\nRunning inference...")
    predictions, probabilities = detector.predict(texts)
    
    for i, text in enumerate(texts):
        pred = predictions[i]
        prob = probabilities[i]
        label = "Anomaly" if pred == 1 else "Normal"
        confidence = prob[pred] * 100
        
        print(f"\nText: {text[:60]}...")
        print(f"  Prediction: {label}")
        print(f"  Confidence: {confidence:.2f}%")
        print(f"  Probabilities: Normal={prob[0]:.4f}, Anomaly={prob[1]:.4f}")


def example_metrics_computation():
    """Example: Compute evaluation metrics."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Computing Metrics")
    print("="*70)
    
    import numpy as np
    
    # Example predictions
    y_true = np.array([0, 0, 0, 1, 1, 0, 1, 0, 0, 1])
    y_pred = np.array([0, 0, 1, 1, 1, 0, 0, 0, 0, 1])
    y_proba = np.array([
        [0.9, 0.1], [0.8, 0.2], [0.4, 0.6], [0.2, 0.8], [0.1, 0.9],
        [0.85, 0.15], [0.6, 0.4], [0.75, 0.25], [0.9, 0.1], [0.15, 0.85]
    ])
    
    # Compute metrics
    calculator = BenchmarkMetrics()
    metrics = calculator.compute_all_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba
    )
    
    print("\nKey Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision_binary']:.4f}")
    print(f"  Recall: {metrics['recall_binary']:.4f}")
    print(f"  F1-Score: {metrics['f1_binary']:.4f}")


def example_complete_workflow():
    """Example: Show complete workflow overview."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Complete Workflow")
    print("="*70)
    
    workflow = """
    
Complete Benchmarking Workflow:

1. Download Data
   └─> python scripts/download_data.py
   
2. Preprocess
   ├─> Parse logs (Drain)
   ├─> Map templates
   ├─> Build sequences
   ├─> Add labels
   ├─> Convert to text
   └─> Split train/test
   
3. Benchmark
   ├─> Load model
   ├─> Run inference
   └─> Compute metrics
   
4. Analyze Results
   └─> python scripts/view_results.py

Or run everything:
   └─> python run_all.py
   
    """
    print(workflow)


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("LOG ANOMALY DETECTION - EXAMPLE USAGE")
    print("="*70)
    
    print("\nThese examples demonstrate key components of the system.")
    print("For full benchmarking, run: python run_all.py")
    
    # Example 1: Log parsing
    example_parse_logs()
    
    # Example 2: Template mapping
    example_template_mapping()
    
    # Example 3: Model inference (commented out by default - requires model download)
    # Uncomment to test model inference
    # example_model_inference()
    
    # Example 4: Metrics
    example_metrics_computation()
    
    # Example 5: Workflow
    example_complete_workflow()
    
    print("\n" + "="*70)
    print("Examples complete!")
    print("="*70)


if __name__ == "__main__":
    main()

