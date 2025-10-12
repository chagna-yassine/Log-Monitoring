"""
View Results

Display benchmark results in a readable format.
"""

import json
import sys
from pathlib import Path


def print_section(title: str, width: int = 70):
    """Print a section header."""
    print("\n" + "="*width)
    print(title.center(width))
    print("="*width)


def print_subsection(title: str, width: int = 70):
    """Print a subsection header."""
    print("\n" + "-"*width)
    print(title)
    print("-"*width)


def format_percentage(value: float) -> str:
    """Format value as percentage."""
    return f"{value*100:.2f}%"


def display_results(results_file: str):
    """
    Display benchmark results from JSON file.
    
    Args:
        results_file: Path to results JSON file
    """
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Header
    print_section("LOG ANOMALY DETECTION BENCHMARK RESULTS")
    
    print(f"\nTimestamp: {results['timestamp']}")
    
    # Model Information
    print_subsection("MODEL INFORMATION")
    model = results['model']
    print(f"Model Name:    {model['model_name']}")
    print(f"Model Type:    {model['model_type']}")
    print(f"Device:        {model['device']}")
    print(f"Num Labels:    {model['num_labels']}")
    print(f"Max Length:    {model['max_length']}")
    print(f"Batch Size:    {model['batch_size']}")
    
    # Dataset Information
    print_subsection("DATASET INFORMATION")
    dataset = results['dataset']
    print(f"Dataset:       {dataset['name']}")
    print(f"Test File:     {dataset['test_file']}")
    print(f"Test Samples:  {dataset['num_samples']:,}")
    
    # Key Metrics
    print_subsection("KEY METRICS")
    metrics = results['metrics']
    
    print(f"\nAccuracy:                {metrics['accuracy']:.4f}")
    
    print(f"\nAnomaly Detection (Binary):")
    print(f"  Precision:             {metrics['precision_binary']:.4f}")
    print(f"  Recall:                {metrics['recall_binary']:.4f}")
    print(f"  F1-Score:              {metrics['f1_binary']:.4f}")
    
    if metrics.get('auc_roc'):
        print(f"  AUC-ROC:               {metrics['auc_roc']:.4f}")
    
    print(f"\nWeighted Averages:")
    print(f"  Precision:             {metrics['precision_weighted']:.4f}")
    print(f"  Recall:                {metrics['recall_weighted']:.4f}")
    print(f"  F1-Score:              {metrics['f1_weighted']:.4f}")
    
    # Confusion Matrix
    print_subsection("CONFUSION MATRIX")
    cm = metrics['confusion_matrix']
    print(f"\n                  Predicted Normal  Predicted Anomaly")
    print(f"Actual Normal     {cm[0][0]:15,}  {cm[0][1]:17,}")
    print(f"Actual Anomaly    {cm[1][0]:15,}  {cm[1][1]:17,}")
    
    if 'true_positives' in metrics:
        print(f"\nDetailed Metrics:")
        print(f"  True Positives:        {metrics['true_positives']:,}")
        print(f"  True Negatives:        {metrics['true_negatives']:,}")
        print(f"  False Positives:       {metrics['false_positives']:,}")
        print(f"  False Negatives:       {metrics['false_negatives']:,}")
        print(f"  Specificity:           {metrics['specificity']:.4f}")
        print(f"  False Positive Rate:   {metrics['false_positive_rate']:.4f}")
        print(f"  False Negative Rate:   {metrics['false_negative_rate']:.4f}")
    
    # Per-Class Metrics
    print_subsection("PER-CLASS METRICS")
    per_class = metrics['per_class']
    for class_name, class_metrics in per_class.items():
        print(f"\n{class_name}:")
        print(f"  Precision:             {class_metrics['precision']:.4f}")
        print(f"  Recall:                {class_metrics['recall']:.4f}")
        print(f"  F1-Score:              {class_metrics['f1_score']:.4f}")
        print(f"  Support:               {class_metrics['support']:,}")
    
    # Dataset Statistics
    print_subsection("DATASET STATISTICS")
    stats = metrics['dataset_stats']
    print(f"\nTotal Samples:           {stats['total_samples']:,}")
    print(f"\nTrue Distribution:")
    print(f"  Normal:                {stats['true_normal_count']:,} ({stats['true_normal_percentage']:.2f}%)")
    print(f"  Anomaly:               {stats['true_anomaly_count']:,} ({stats['true_anomaly_percentage']:.2f}%)")
    print(f"\nPredicted Distribution:")
    print(f"  Normal:                {stats['predicted_normal_count']:,} ({stats['predicted_normal_percentage']:.2f}%)")
    print(f"  Anomaly:               {stats['predicted_anomaly_count']:,} ({stats['predicted_anomaly_percentage']:.2f}%)")
    
    # Performance Metrics
    if 'inference_time_seconds' in metrics:
        print_subsection("PERFORMANCE METRICS")
        print(f"Inference Time:          {metrics['inference_time_seconds']:.2f} seconds")
        print(f"Throughput:              {metrics['throughput_samples_per_second']:.2f} samples/second")
        print(f"Latency:                 {metrics['latency_ms_per_sample']:.4f} ms/sample")
    
    # Footer
    print("\n" + "="*70)


def main():
    """Main function."""
    results_file = Path("results/benchmark_results.json")
    
    if len(sys.argv) > 1:
        results_file = Path(sys.argv[1])
    
    if not results_file.exists():
        print(f"Error: Results file not found: {results_file}")
        print("\nUsage: python scripts/view_results.py [results_file.json]")
        print("Default: results/benchmark_results.json")
        sys.exit(1)
    
    display_results(str(results_file))


if __name__ == "__main__":
    main()

