"""
Results Comparison Script

Compares benchmark results between HDFS and BGL datasets.
"""

import json
import sys
from pathlib import Path


def load_results(file_path: str) -> dict:
    """Load benchmark results from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def print_section(title: str, width: int = 70):
    """Print a section header."""
    print("\n" + "="*width)
    print(title.center(width))
    print("="*width)


def print_metric_comparison(metric_name: str, hdfs_value: float, bgl_value: float):
    """Print a metric comparison."""
    diff = bgl_value - hdfs_value
    diff_percent = (diff / hdfs_value * 100) if hdfs_value != 0 else 0
    
    print(f"{metric_name:25s} | {hdfs_value:8.4f} | {bgl_value:8.4f} | {diff:8.4f} ({diff_percent:+6.1f}%)")


def main():
    """Main function to compare results."""
    print("="*70)
    print("BENCHMARK RESULTS COMPARISON")
    print("HDFS vs BGL Dataset Performance")
    print("="*70)
    
    # Default file paths
    hdfs_results = Path("results/benchmark_results.json")
    bgl_results = Path("results/bgl_benchmark_results.json")
    
    # Check if custom paths provided
    if len(sys.argv) >= 3:
        hdfs_results = Path(sys.argv[1])
        bgl_results = Path(sys.argv[2])
    
    # Verify files exist
    if not hdfs_results.exists():
        print(f"Error: HDFS results file not found: {hdfs_results}")
        print("Run HDFS benchmark first: python scripts/run_dataset.py")
        sys.exit(1)
    
    if not bgl_results.exists():
        print(f"Error: BGL results file not found: {bgl_results}")
        print("Run BGL benchmark first: python scripts/run_dataset.py")
        sys.exit(1)
    
    # Load results
    print(f"\nLoading HDFS results from: {hdfs_results}")
    hdfs_data = load_results(str(hdfs_results))
    
    print(f"Loading BGL results from: {bgl_results}")
    bgl_data = load_results(str(bgl_results))
    
    # Extract metrics
    hdfs_metrics = hdfs_data['metrics']
    bgl_metrics = bgl_data['metrics']
    
    # Dataset information
    print_section("DATASET INFORMATION")
    print(f"HDFS Dataset:")
    print(f"  Name: {hdfs_data['dataset']['name']}")
    print(f"  Test Samples: {hdfs_data['dataset']['num_samples']:,}")
    print(f"  Timestamp: {hdfs_data['timestamp']}")
    
    print(f"\nBGL Dataset:")
    print(f"  Name: {bgl_data['dataset']['name']}")
    print(f"  Test Samples: {bgl_data['dataset']['num_samples']:,}")
    print(f"  Timestamp: {bgl_data['timestamp']}")
    
    # Model information
    print_section("MODEL INFORMATION")
    print(f"Model: {hdfs_data['model']['model_name']}")
    print(f"Device: {hdfs_data['model']['device']}")
    print(f"Batch Size: {hdfs_data['model']['batch_size']}")
    print(f"Max Length: {hdfs_data['model']['max_length']}")
    
    # Key metrics comparison
    print_section("KEY METRICS COMPARISON")
    print(f"{'Metric':25s} | {'HDFS':8s} | {'BGL':8s} | {'Difference':15s}")
    print("-" * 70)
    
    print_metric_comparison("Accuracy", hdfs_metrics['accuracy'], bgl_metrics['accuracy'])
    print_metric_comparison("Precision (Binary)", hdfs_metrics['precision_binary'], bgl_metrics['precision_binary'])
    print_metric_comparison("Recall (Binary)", hdfs_metrics['recall_binary'], bgl_metrics['recall_binary'])
    print_metric_comparison("F1-Score (Binary)", hdfs_metrics['f1_binary'], bgl_metrics['f1_binary'])
    
    if 'auc_roc' in hdfs_metrics and 'auc_roc' in bgl_metrics:
        if hdfs_metrics['auc_roc'] is not None and bgl_metrics['auc_roc'] is not None:
            print_metric_comparison("AUC-ROC", hdfs_metrics['auc_roc'], bgl_metrics['auc_roc'])
    
    # Weighted metrics
    print_section("WEIGHTED METRICS COMPARISON")
    print(f"{'Metric':25s} | {'HDFS':8s} | {'BGL':8s} | {'Difference':15s}")
    print("-" * 70)
    
    print_metric_comparison("Precision (Weighted)", hdfs_metrics['precision_weighted'], bgl_metrics['precision_weighted'])
    print_metric_comparison("Recall (Weighted)", hdfs_metrics['recall_weighted'], bgl_metrics['recall_weighted'])
    print_metric_comparison("F1-Score (Weighted)", hdfs_metrics['f1_weighted'], bgl_metrics['f1_weighted'])
    
    # Performance metrics
    print_section("PERFORMANCE METRICS COMPARISON")
    print(f"{'Metric':25s} | {'HDFS':8s} | {'BGL':8s} | {'Difference':15s}")
    print("-" * 70)
    
    hdfs_time = hdfs_metrics['inference_time_seconds']
    bgl_time = bgl_metrics['inference_time_seconds']
    print_metric_comparison("Inference Time (s)", hdfs_time, bgl_time)
    
    hdfs_throughput = hdfs_metrics['throughput_samples_per_second']
    bgl_throughput = bgl_metrics['throughput_samples_per_second']
    print_metric_comparison("Throughput (samples/s)", hdfs_throughput, bgl_throughput)
    
    # Dataset statistics
    print_section("DATASET STATISTICS")
    
    hdfs_stats = hdfs_metrics['dataset_stats']
    bgl_stats = bgl_metrics['dataset_stats']
    
    print(f"HDFS Dataset:")
    print(f"  Total Samples: {hdfs_stats['total_samples']:,}")
    print(f"  Normal: {hdfs_stats['true_normal_count']:,} ({hdfs_stats['true_normal_percentage']:.2f}%)")
    print(f"  Anomaly: {hdfs_stats['true_anomaly_count']:,} ({hdfs_stats['true_anomaly_percentage']:.2f}%)")
    
    print(f"\nBGL Dataset:")
    print(f"  Total Samples: {bgl_stats['total_samples']:,}")
    print(f"  Normal: {bgl_stats['true_normal_count']:,} ({bgl_stats['true_normal_percentage']:.2f}%)")
    print(f"  Anomaly: {bgl_stats['true_anomaly_count']:,} ({bgl_stats['true_anomaly_percentage']:.2f}%)")
    
    # Confusion matrix comparison
    print_section("CONFUSION MATRIX COMPARISON")
    
    print("HDFS Confusion Matrix:")
    hdfs_cm = hdfs_metrics['confusion_matrix']
    print(f"                  Predicted Normal  Predicted Anomaly")
    print(f"Actual Normal     {hdfs_cm[0][0]:15,}  {hdfs_cm[0][1]:17,}")
    print(f"Actual Anomaly    {hdfs_cm[1][0]:15,}  {hdfs_cm[1][1]:17,}")
    
    print("\nBGL Confusion Matrix:")
    bgl_cm = bgl_metrics['confusion_matrix']
    print(f"                  Predicted Normal  Predicted Anomaly")
    print(f"Actual Normal     {bgl_cm[0][0]:15,}  {bgl_cm[0][1]:17,}")
    print(f"Actual Anomaly    {bgl_cm[1][0]:15,}  {bgl_cm[1][1]:17,}")
    
    # Summary
    print_section("SUMMARY")
    
    # Find best performing dataset for each metric
    metrics_to_compare = [
        ('Accuracy', hdfs_metrics['accuracy'], bgl_metrics['accuracy']),
        ('F1-Score (Binary)', hdfs_metrics['f1_binary'], bgl_metrics['f1_binary']),
        ('AUC-ROC', hdfs_metrics.get('auc_roc', 0), bgl_metrics.get('auc_roc', 0))
    ]
    
    hdfs_wins = 0
    bgl_wins = 0
    
    for metric_name, hdfs_val, bgl_val in metrics_to_compare:
        if hdfs_val > bgl_val:
            hdfs_wins += 1
            winner = "HDFS"
        elif bgl_val > hdfs_val:
            bgl_wins += 1
            winner = "BGL"
        else:
            winner = "Tie"
        
        print(f"{metric_name}: {winner} wins")
    
    print(f"\nOverall: HDFS wins {hdfs_wins}, BGL wins {bgl_wins}")
    
    if hdfs_wins > bgl_wins:
        print("HDFS dataset shows better overall performance")
    elif bgl_wins > hdfs_wins:
        print("BGL dataset shows better overall performance")
    else:
        print("Both datasets show similar performance")
    
    print("\n" + "="*70)
    print("Comparison complete!")
    print("="*70)


if __name__ == "__main__":
    main()
