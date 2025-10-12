"""
Benchmark Metrics

Comprehensive evaluation metrics for log anomaly detection.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from typing import Dict, List, Tuple
import time


class BenchmarkMetrics:
    """
    Computes comprehensive metrics for binary classification (anomaly detection).
    """
    
    def __init__(self):
        self.label_names = {0: 'Normal', 1: 'Anomaly'}
        
    def compute_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray = None,
        inference_time: float = None,
        num_samples: int = None
    ) -> Dict:
        """
        Compute all benchmark metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (for AUC-ROC)
            inference_time: Total inference time in seconds
            num_samples: Number of samples processed
            
        Returns:
            Dictionary with all metrics
        """
        print("\n" + "="*60)
        print("COMPUTING BENCHMARK METRICS")
        print("="*60)
        
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Binary classification metrics
        metrics['precision_binary'] = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        metrics['recall_binary'] = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        metrics['f1_binary'] = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        
        # Weighted metrics (account for class imbalance)
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Macro metrics (treat all classes equally)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Micro metrics (global)
        metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
        
        # AUC-ROC (requires probabilities)
        if y_proba is not None:
            try:
                # Use probability of positive class (anomaly)
                if len(y_proba.shape) > 1 and y_proba.shape[1] == 2:
                    y_score = y_proba[:, 1]
                else:
                    y_score = y_proba
                metrics['auc_roc'] = roc_auc_score(y_true, y_score)
            except Exception as e:
                print(f"Warning: Could not compute AUC-ROC: {e}")
                metrics['auc_roc'] = None
        else:
            metrics['auc_roc'] = None
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Confusion matrix elements
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['true_negatives'] = int(tn)
            metrics['false_positives'] = int(fp)
            metrics['false_negatives'] = int(fn)
            metrics['true_positives'] = int(tp)
            
            # Additional metrics
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
            metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Per-class metrics
        per_class = self._compute_per_class_metrics(y_true, y_pred)
        metrics['per_class'] = per_class
        
        # Dataset statistics
        metrics['dataset_stats'] = self._compute_dataset_stats(y_true, y_pred)
        
        # Performance metrics
        if inference_time is not None and num_samples is not None:
            metrics['inference_time_seconds'] = inference_time
            metrics['throughput_samples_per_second'] = num_samples / inference_time if inference_time > 0 else 0
            metrics['latency_ms_per_sample'] = (inference_time / num_samples) * 1000 if num_samples > 0 else 0
        
        # Print summary
        self.print_metrics_summary(metrics)
        
        return metrics
    
    def _compute_per_class_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict:
        """
        Compute metrics for each class separately.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary with per-class metrics
        """
        per_class = {}
        
        for label in [0, 1]:
            label_name = self.label_names[label]
            
            # Binary metrics treating this class as positive
            precision = precision_score(
                y_true, y_pred,
                pos_label=label,
                average='binary',
                zero_division=0
            )
            recall = recall_score(
                y_true, y_pred,
                pos_label=label,
                average='binary',
                zero_division=0
            )
            f1 = f1_score(
                y_true, y_pred,
                pos_label=label,
                average='binary',
                zero_division=0
            )
            
            # Support (number of samples)
            support = int(np.sum(y_true == label))
            
            per_class[label_name] = {
                'label': int(label),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'support': support
            }
        
        return per_class
    
    def _compute_dataset_stats(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict:
        """
        Compute dataset statistics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary with dataset statistics
        """
        total = len(y_true)
        
        true_normal = int(np.sum(y_true == 0))
        true_anomaly = int(np.sum(y_true == 1))
        
        pred_normal = int(np.sum(y_pred == 0))
        pred_anomaly = int(np.sum(y_pred == 1))
        
        return {
            'total_samples': total,
            'true_normal_count': true_normal,
            'true_anomaly_count': true_anomaly,
            'true_normal_percentage': (true_normal / total) * 100 if total > 0 else 0,
            'true_anomaly_percentage': (true_anomaly / total) * 100 if total > 0 else 0,
            'predicted_normal_count': pred_normal,
            'predicted_anomaly_count': pred_anomaly,
            'predicted_normal_percentage': (pred_normal / total) * 100 if total > 0 else 0,
            'predicted_anomaly_percentage': (pred_anomaly / total) * 100 if total > 0 else 0
        }
    
    def print_metrics_summary(self, metrics: Dict) -> None:
        """
        Print a formatted summary of metrics.
        
        Args:
            metrics: Dictionary with computed metrics
        """
        print("\n" + "-"*60)
        print("OVERALL METRICS")
        print("-"*60)
        print(f"Accuracy:              {metrics['accuracy']:.4f}")
        print()
        
        print("Binary Classification (Anomaly Detection):")
        print(f"  Precision:           {metrics['precision_binary']:.4f}")
        print(f"  Recall:              {metrics['recall_binary']:.4f}")
        print(f"  F1-Score:            {metrics['f1_binary']:.4f}")
        
        if metrics.get('auc_roc') is not None:
            print(f"  AUC-ROC:             {metrics['auc_roc']:.4f}")
        print()
        
        print("Weighted Averages:")
        print(f"  Precision:           {metrics['precision_weighted']:.4f}")
        print(f"  Recall:              {metrics['recall_weighted']:.4f}")
        print(f"  F1-Score:            {metrics['f1_weighted']:.4f}")
        print()
        
        print("Macro Averages:")
        print(f"  Precision:           {metrics['precision_macro']:.4f}")
        print(f"  Recall:              {metrics['recall_macro']:.4f}")
        print(f"  F1-Score:            {metrics['f1_macro']:.4f}")
        print()
        
        print("-"*60)
        print("CONFUSION MATRIX")
        print("-"*60)
        cm = np.array(metrics['confusion_matrix'])
        print(f"                  Predicted Normal  Predicted Anomaly")
        print(f"Actual Normal     {cm[0][0]:15d}  {cm[0][1]:17d}")
        print(f"Actual Anomaly    {cm[1][0]:15d}  {cm[1][1]:17d}")
        print()
        
        if 'true_positives' in metrics:
            print("Confusion Matrix Details:")
            print(f"  True Positives:      {metrics['true_positives']:,}")
            print(f"  True Negatives:      {metrics['true_negatives']:,}")
            print(f"  False Positives:     {metrics['false_positives']:,}")
            print(f"  False Negatives:     {metrics['false_negatives']:,}")
            print()
            print(f"  Specificity:         {metrics['specificity']:.4f}")
            print(f"  False Positive Rate: {metrics['false_positive_rate']:.4f}")
            print(f"  False Negative Rate: {metrics['false_negative_rate']:.4f}")
            print()
        
        print("-"*60)
        print("PER-CLASS METRICS")
        print("-"*60)
        for class_name, class_metrics in metrics['per_class'].items():
            print(f"{class_name}:")
            print(f"  Precision:           {class_metrics['precision']:.4f}")
            print(f"  Recall:              {class_metrics['recall']:.4f}")
            print(f"  F1-Score:            {class_metrics['f1_score']:.4f}")
            print(f"  Support:             {class_metrics['support']:,}")
            print()
        
        print("-"*60)
        print("DATASET STATISTICS")
        print("-"*60)
        stats = metrics['dataset_stats']
        print(f"Total Samples:         {stats['total_samples']:,}")
        print()
        print("True Distribution:")
        print(f"  Normal:              {stats['true_normal_count']:,} ({stats['true_normal_percentage']:.2f}%)")
        print(f"  Anomaly:             {stats['true_anomaly_count']:,} ({stats['true_anomaly_percentage']:.2f}%)")
        print()
        print("Predicted Distribution:")
        print(f"  Normal:              {stats['predicted_normal_count']:,} ({stats['predicted_normal_percentage']:.2f}%)")
        print(f"  Anomaly:             {stats['predicted_anomaly_count']:,} ({stats['predicted_anomaly_percentage']:.2f}%)")
        print()
        
        if 'inference_time_seconds' in metrics:
            print("-"*60)
            print("PERFORMANCE METRICS")
            print("-"*60)
            print(f"Inference Time:        {metrics['inference_time_seconds']:.2f} seconds")
            print(f"Throughput:            {metrics['throughput_samples_per_second']:.2f} samples/second")
            print(f"Latency:               {metrics['latency_ms_per_sample']:.4f} ms/sample")
            print()
        
        print("="*60)

