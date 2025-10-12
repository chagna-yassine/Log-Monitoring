"""
Benchmark Script

Runs the complete benchmarking pipeline for log anomaly detection.
"""

import os
import sys
from pathlib import Path
import yaml
import json
import pandas as pd
import numpy as np
from datetime import datetime
import time

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model import LogAnomalyDetector
from evaluation import BenchmarkMetrics


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_results(results: dict, output_path: str) -> None:
    """
    Save benchmark results to JSON file.
    
    Args:
        results: Results dictionary
        output_path: Path to save JSON file
    """
    # Convert numpy types to Python types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        return obj
    
    results = convert_types(results)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_path}")


def main():
    """Main function to run benchmarking."""
    print("="*70)
    print("LOG ANOMALY DETECTION MODEL BENCHMARKING")
    print("="*70)
    
    # Load configuration
    print("\nLoading configuration...")
    config = load_config()
    
    model_config = config['model']
    output_config = config['output']
    
    # Setup paths
    output_path = Path(output_config['base_path'])
    test_csv = output_path / "hdfs_test.csv"
    
    results_dir = Path(output_config['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Verify test file exists
    if not test_csv.exists():
        print(f"Error: Test file not found: {test_csv}")
        print("Please run preprocess.py first.")
        sys.exit(1)
    
    print(f"Test file: {test_csv}")
    
    # Load test data
    print("\n" + "="*70)
    print("LOADING TEST DATA")
    print("="*70)
    
    test_df = pd.read_csv(test_csv)
    print(f"Loaded {len(test_df):,} test samples")
    
    # Extract texts and labels
    texts = test_df['Text'].tolist()
    y_true = test_df['Label'].values
    
    # Label distribution
    unique, counts = np.unique(y_true, return_counts=True)
    print("\nLabel distribution:")
    for label, count in zip(unique, counts):
        label_name = "Normal" if label == 0 else "Anomaly"
        percentage = (count / len(y_true)) * 100
        print(f"  {label_name} ({label}): {count:,} ({percentage:.2f}%)")
    
    # Load model
    print("\n" + "="*70)
    print("LOADING MODEL")
    print("="*70)
    
    detector = LogAnomalyDetector(
        model_name=model_config['name'],
        max_length=model_config['max_length'],
        batch_size=model_config['batch_size'],
        device=model_config['device']
    )
    
    model_info = detector.get_model_info()
    print("\nModel information:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    # Run inference
    print("\n" + "="*70)
    print("RUNNING INFERENCE")
    print("="*70)
    
    start_time = time.time()
    y_pred, y_proba = detector.predict(texts, return_probabilities=True)
    inference_time = time.time() - start_time
    
    print(f"\nInference complete!")
    print(f"  Time: {inference_time:.2f} seconds")
    print(f"  Throughput: {len(texts) / inference_time:.2f} samples/second")
    
    # Compute metrics
    print("\n" + "="*70)
    print("COMPUTING METRICS")
    print("="*70)
    
    metrics_calculator = BenchmarkMetrics()
    metrics = metrics_calculator.compute_all_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        inference_time=inference_time,
        num_samples=len(texts)
    )
    
    # Prepare results
    timestamp = datetime.now().isoformat()
    
    results = {
        'timestamp': timestamp,
        'model': model_info,
        'dataset': {
            'name': config['dataset']['name'],
            'test_file': str(test_csv),
            'num_samples': len(test_df)
        },
        'metrics': metrics,
        'configuration': {
            'train_ratio': config['preprocessing']['train_ratio'],
            'random_seed': config['preprocessing']['random_seed'],
            'batch_size': model_config['batch_size'],
            'max_length': model_config['max_length']
        }
    }
    
    # Save results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    results_file = results_dir / "benchmark_results.json"
    save_results(results, str(results_file))
    
    # Save predictions
    predictions_df = test_df.copy()
    predictions_df['Predicted'] = y_pred
    predictions_df['Probability_Normal'] = y_proba[:, 0]
    predictions_df['Probability_Anomaly'] = y_proba[:, 1]
    predictions_df['Correct'] = (y_pred == y_true).astype(int)
    
    predictions_file = results_dir / "predictions.csv"
    predictions_df.to_csv(predictions_file, index=False)
    print(f"Predictions saved to: {predictions_file}")
    
    # Final summary
    print("\n" + "="*70)
    print("BENCHMARKING COMPLETE!")
    print("="*70)
    
    print("\n📊 KEY RESULTS:")
    print(f"  Accuracy:              {metrics['accuracy']:.4f}")
    print(f"  Precision (Anomaly):   {metrics['precision_binary']:.4f}")
    print(f"  Recall (Anomaly):      {metrics['recall_binary']:.4f}")
    print(f"  F1-Score (Anomaly):    {metrics['f1_binary']:.4f}")
    if metrics.get('auc_roc'):
        print(f"  AUC-ROC:               {metrics['auc_roc']:.4f}")
    print(f"\n⚡ PERFORMANCE:")
    print(f"  Inference Time:        {inference_time:.2f} seconds")
    print(f"  Throughput:            {metrics['throughput_samples_per_second']:.2f} samples/sec")
    
    print("\n" + "="*70)
    print(f"Results saved to: {results_dir.absolute()}")
    print("="*70)


if __name__ == "__main__":
    main()

