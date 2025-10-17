"""
AIT-LDS Benchmarking

Runs comprehensive benchmarking on AIT-LDS test data.
"""

import pandas as pd
import yaml
from pathlib import Path
import sys
import json
import time

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

from model.inference import LogAnomalyDetector
from evaluation.metrics import BenchmarkMetrics


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Run AIT-LDS benchmarking."""
    print("="*70)
    print("AIT-LDS LOG ANOMALY DETECTION BENCHMARKING")
    print("="*70)

    # Load configuration
    config = load_config()
    model_config = config['model']
    ait_config = config['ait_dataset']
    output_config = config['output']['ait']
    results_dir = Path(config['output']['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)

    # Setup paths
    output_path = Path(output_config['base_path'])
    test_file = output_path / "ait_test.csv"

    if not test_file.exists():
        print(f"Error: AIT-LDS test file not found: {test_file}")
        print("Please run: python scripts/preprocess_ait.py")
        sys.exit(1)

    print(f"Loading AIT-LDS test data from: {test_file}")
    test_df = pd.read_csv(test_file)
    
    if test_df.empty:
        print("Error: AIT-LDS test DataFrame is empty.")
        sys.exit(1)

    # Prepare data
    if 'TextSequence' in test_df.columns:
        texts = test_df['TextSequence'].tolist()
    elif 'Text' in test_df.columns:
        texts = test_df['Text'].tolist()
    else:
        print("Error: No text column found in test data")
        print(f"Available columns: {test_df.columns.tolist()}")
        sys.exit(1)
    
    labels = test_df['Label'].tolist()

    print(f"Loaded {len(texts):,} AIT-LDS test samples")
    print(f"Dataset: {ait_config['selected_dataset']}")
    print(f"Anomaly rate: {test_df['Label'].mean():.2%} (1=Attack)")
    print(f"Normal samples: {(test_df['Label'] == 0).sum():,}")
    print(f"Attack samples: {(test_df['Label'] == 1).sum():,}")

    # Initialize model
    print(f"\n" + "="*70)
    print(f"STEP 1: MODEL INFERENCE ({model_config['name']})")
    print("="*70)
    
    detector = LogAnomalyDetector(
        model_name=model_config['name'],
        max_length=model_config['max_length'],
        batch_size=model_config['batch_size'],
        device=model_config['device']
    )

    start_time = time.time()
    predictions, probabilities = detector.predict(texts)
    end_time = time.time()
    inference_time = end_time - start_time

    print(f"Inference complete in {inference_time:.2f} seconds.")
    print(f"Throughput: {len(texts) / inference_time:.2f} samples/second.")

    # Initialize metrics calculator
    print(f"\n" + "="*70)
    print("STEP 2: EVALUATION METRICS")
    print("="*70)
    
    metrics_calculator = BenchmarkMetrics()

    # Compute all metrics
    results = metrics_calculator.compute_all_metrics(
        y_true=labels,
        y_pred=predictions,
        y_proba=probabilities[:, 1],  # Probability of positive class (Attack)
        inference_time=inference_time,
        num_samples=len(texts)
    )

    # Add model and dataset info to results
    results['model'] = model_config
    results['dataset'] = {
        'name': f"AIT-LDS_{ait_config['selected_dataset']}",
        'test_file': str(test_file),
        'num_samples': len(texts),
        'dataset_info': {
            'selected_dataset': ait_config['selected_dataset'],
            'available_datasets': [d['name'] for d in ait_config['datasets']],
            'log_types': ait_config['log_types'],
            'attack_types': ait_config['attack_types']
        }
    }
    results['timestamp'] = pd.Timestamp.now().isoformat()
    results['configuration'] = {
        'train_ratio': config['preprocessing']['train_ratio'],
        'random_seed': config['preprocessing']['random_seed'],
        'batch_size': model_config['batch_size'],
        'max_length': model_config['max_length']
    }

    # Save results
    results_file = results_dir / f"ait_{ait_config['selected_dataset']}_benchmark_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Benchmark results saved to: {results_file}")

    # Save predictions for detailed analysis
    predictions_df = pd.DataFrame({
        'TextSequence': texts,
        'TrueLabel': labels,
        'PredictedLabel': predictions,
        'AttackProbability': probabilities[:, 1]
    })
    
    # Add additional columns if available
    if 'SessionId' in test_df.columns:
        predictions_df['SessionId'] = test_df['SessionId']
    if 'IsAttack' in test_df.columns:
        predictions_df['IsAttack'] = test_df['IsAttack']
    if 'AttackLabels' in test_df.columns:
        predictions_df['AttackLabels'] = test_df['AttackLabels']
    
    predictions_file = results_dir / f"ait_{ait_config['selected_dataset']}_predictions.csv"
    predictions_df.to_csv(predictions_file, index=False)
    print(f"Predictions saved to: {predictions_file}")

    # Display key results
    print(f"\n" + "="*70)
    print("AIT-LDS BENCHMARK RESULTS")
    print("="*70)
    
    metrics = results['metrics']
    print(f"\nKey Performance Metrics:")
    print(f"  Accuracy:        {metrics['accuracy']:.4f}")
    print(f"  Precision:       {metrics['precision_binary']:.4f}")
    print(f"  Recall:          {metrics['recall_binary']:.4f}")
    print(f"  F1-Score:        {metrics['f1_binary']:.4f}")
    print(f"  AUC-ROC:         {metrics['auc_roc']:.4f}")
    
    print(f"\nPerformance Metrics:")
    print(f"  Inference Time:  {metrics['inference_time_seconds']:.2f} seconds")
    print(f"  Throughput:      {metrics['throughput_samples_per_second']:.2f} samples/second")
    
    if 'per_class' in metrics:
        print(f"\nPer-Class Performance:")
        for class_name, class_metrics in metrics['per_class'].items():
            print(f"  {class_name}:")
            print(f"    Precision: {class_metrics['precision']:.4f}")
            print(f"    Recall:    {class_metrics['recall']:.4f}")
            print(f"    F1-Score:  {class_metrics['f1_score']:.4f}")
            print(f"    Support:   {class_metrics['support']:,}")
    
    # Dataset-specific analysis
    print(f"\nDataset Information:")
    print(f"  Selected Dataset: {ait_config['selected_dataset']}")
    print(f"  Available Datasets: {', '.join([d['name'] for d in ait_config['datasets']])}")
    print(f"  Log Types: {', '.join(ait_config['log_types'])}")
    print(f"  Attack Types: {', '.join(ait_config['attack_types'])}")

    print("\n" + "="*70)
    print("AIT-LDS BENCHMARKING COMPLETE!")
    print("="*70)
    
    print(f"\nFiles generated:")
    print(f"  Results: {results_file}")
    print(f"  Predictions: {predictions_file}")
    
    print(f"\nTo benchmark other AIT-LDS datasets:")
    print(f"  1. Edit config.yaml: ait_dataset.selected_dataset")
    print(f"  2. Run: python scripts/download_data_ait.py")
    print(f"  3. Run: python scripts/preprocess_ait.py")
    print(f"  4. Run: python scripts/benchmark_ait.py")


if __name__ == "__main__":
    main()
