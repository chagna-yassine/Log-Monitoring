# Log Anomaly Detection Model Benchmarking Documentation

## Overview

This document provides comprehensive technical documentation for the benchmarking system used to evaluate the `Dumi2025/log-anomaly-detection-model-roberta` model on log anomaly detection tasks. The system follows the LogBERT methodology for fair comparison and evaluation.

## Model Information

### Model Architecture
- **Model Name**: `Dumi2025/log-anomaly-detection-model-roberta`
- **Architecture**: RoBERTa-base (Transformer-based)
- **Task**: Binary sequence classification (Normal/Anomaly)
- **Source**: Hugging Face Model Hub
- **Fine-tuning**: Pre-trained on log anomaly detection task

### Model Configuration
```yaml
model:
  name: "Dumi2025/log-anomaly-detection-model-roberta"
  max_length: 512      # Maximum tokenization length
  batch_size: 32       # Inference batch size
  device: "cuda"       # Device selection (CUDA/CPU)
```

## Input Format

### Model Input Type
The model accepts **text sequences** as input, where each sequence represents a log block's event history.

### Input Format Details
- **Type**: List of strings (`List[str]`)
- **Content**: Space-separated event templates joined with " | " separator
- **Length**: Variable length sequences (truncated to max 512 tokens)
- **Encoding**: Tokenized using RoBERTa tokenizer

### Example Input Sequences
```python
# Normal sequence example
"Receiving block | PacketResponder for block | Verification succeeded | Block written"

# Anomaly sequence example  
"Connection timeout | Failed to read | Error occurred | Connection lost | Retry failed"
```

### Preprocessing Pipeline
The input text is generated through a 6-step preprocessing pipeline:

1. **Log Parsing**: Raw logs → Structured format using Drain algorithm
2. **Template Mapping**: Event templates → Numerical IDs (1, 2, 3, ...)
3. **Sequence Building**: Group events by block ID → Create sequences
4. **Label Assignment**: Map block IDs to Normal (0) or Anomaly (1) labels
5. **Text Conversion**: Numerical sequences → Readable text templates
6. **Data Splitting**: 70% normal blocks for training, 30% normal + all anomalous for testing

### Text Conversion Process
```python
# Numerical sequence: "1 5 12 8 3"
# Template mapping: 1→"Receiving block", 5→"PacketResponder", etc.
# Final text: "Receiving block | PacketResponder | Verification succeeded | Block written | Data transfer"
```

## Output Format

### Model Output Type
The model produces two types of outputs:

1. **Predictions**: Binary classification labels
2. **Probabilities**: Confidence scores for each class

### Output Format Details
- **Predictions**: `numpy.ndarray` of shape `(n_samples,)` with values `[0, 1]`
  - `0` = Normal
  - `1` = Anomaly
- **Probabilities**: `numpy.ndarray` of shape `(n_samples, 2)` with values `[0.0, 1.0]`
  - `[prob_normal, prob_anomaly]` for each sample
  - Probabilities sum to 1.0

### Example Output
```python
# Input: ["Receiving block | PacketResponder", "Connection timeout | Failed to read"]
# Predictions: [0, 1]  # Normal, Anomaly
# Probabilities: [[0.95, 0.05], [0.12, 0.88]]  # High confidence in both cases
```

## Benchmarking Process

### Technical Implementation
The benchmarking process follows these steps:

1. **Data Loading**: Load preprocessed test dataset
2. **Model Initialization**: Load RoBERTa model and tokenizer
3. **Batch Processing**: Process samples in configurable batches
4. **Inference**: Run forward pass through the model
5. **Metrics Computation**: Calculate comprehensive evaluation metrics
6. **Results Storage**: Save results and predictions to files

### Performance Optimization
- **Batch Processing**: Configurable batch size (default: 32)
- **Device Selection**: Automatic CUDA/CPU detection
- **Memory Management**: Efficient tensor operations
- **Progress Tracking**: Real-time inference progress

## Evaluation Metrics

The system computes comprehensive metrics for binary classification (anomaly detection):

### 1. Basic Classification Metrics

#### Accuracy
- **Formula**: `(TP + TN) / (TP + TN + FP + FN)`
- **Range**: [0, 1]
- **Interpretation**: Overall classification correctness
- **Use Case**: General performance indicator

#### Precision (Binary)
- **Formula**: `TP / (TP + FP)`
- **Range**: [0, 1]
- **Interpretation**: Of all predicted anomalies, how many are actually anomalous
- **Use Case**: Measures false alarm rate

#### Recall (Binary)
- **Formula**: `TP / (TP + FN)`
- **Range**: [0, 1]
- **Interpretation**: Of all actual anomalies, how many are correctly detected
- **Use Case**: Measures detection rate

#### F1-Score (Binary)
- **Formula**: `2 * (Precision * Recall) / (Precision + Recall)`
- **Range**: [0, 1]
- **Interpretation**: Harmonic mean of precision and recall
- **Use Case**: Balanced performance metric for imbalanced data

### 2. Averaging Strategies

#### Weighted Metrics
- **Purpose**: Account for class imbalance
- **Calculation**: Weight metrics by class support
- **Use Case**: When class distribution matters

#### Macro Metrics
- **Purpose**: Treat all classes equally
- **Calculation**: Average metrics across classes
- **Use Case**: When both classes are equally important

#### Micro Metrics
- **Purpose**: Global aggregation
- **Calculation**: Aggregate all TP, TN, FP, FN first
- **Use Case**: Overall system performance

### 3. Advanced Metrics

#### AUC-ROC (Area Under ROC Curve)
- **Range**: [0, 1]
- **Interpretation**: Discrimination ability across all thresholds
- **Calculation**: Uses probability scores for positive class
- **Use Case**: Model ranking and threshold-independent evaluation

#### Confusion Matrix Elements
- **True Positives (TP)**: Correctly predicted anomalies
- **True Negatives (TN)**: Correctly predicted normal
- **False Positives (FP)**: Incorrectly predicted anomalies (false alarms)
- **False Negatives (FN)**: Missed anomalies

#### Derived Metrics
- **Specificity**: `TN / (TN + FP)` - True negative rate
- **False Positive Rate**: `FP / (FP + TN)` - False alarm rate
- **False Negative Rate**: `FN / (FN + TP)` - Miss rate

### 4. Per-Class Metrics
Individual metrics computed for each class (Normal/Anomaly):
- Precision, Recall, F1-Score for each class
- Support (number of samples per class)

### 5. Performance Metrics
- **Inference Time**: Total processing time in seconds
- **Throughput**: Samples processed per second
- **Latency**: Average time per sample in milliseconds

### 6. Dataset Statistics
- **Total Samples**: Number of test samples
- **Class Distribution**: Count and percentage of each class
- **Prediction Distribution**: Model's prediction distribution

## Success Criteria

Based on LogBERT methodology, good performance indicators are:

- **F1-Score (Anomaly)**: > 0.90
- **Precision (Anomaly)**: > 0.95
- **Recall (Anomaly)**: > 0.85
- **AUC-ROC**: > 0.95

## Benchmarking Scripts

### Main Benchmarking Script
**File**: `scripts/benchmark.py`
- Loads preprocessed test data
- Initializes model with configuration
- Runs inference with timing
- Computes all metrics
- Saves results and predictions

### Usage
```bash
python scripts/benchmark.py
```

### Output Files
- `results/benchmark_results.json`: Complete metrics and configuration
- `results/predictions.csv`: Individual predictions with probabilities

## Configuration

### Model Configuration
```yaml
model:
  name: "Dumi2025/log-anomaly-detection-model-roberta"
  max_length: 512
  batch_size: 32
  device: "cuda"
```

### Evaluation Configuration
The metrics computation is automatic and comprehensive, requiring no additional configuration.

## Results Interpretation

### Key Metrics Priority
1. **F1-Score (Anomaly)**: Primary metric for imbalanced data
2. **Recall (Anomaly)**: Critical for anomaly detection (minimize missed anomalies)
3. **Precision (Anomaly)**: Important for reducing false alarms
4. **AUC-ROC**: Overall discrimination ability

### Performance Considerations
- **Throughput**: Higher is better for production deployment
- **Latency**: Lower is better for real-time applications
- **Memory Usage**: Depends on batch size and sequence length

## Technical Implementation Details

### Model Loading
```python
detector = LogAnomalyDetector(
    model_name="Dumi2025/log-anomaly-detection-model-roberta",
    max_length=512,
    batch_size=32,
    device="cuda"
)
```

### Inference Process
```python
predictions, probabilities = detector.predict(texts, return_probabilities=True)
```

### Metrics Computation
```python
metrics_calculator = BenchmarkMetrics()
metrics = metrics_calculator.compute_all_metrics(
    y_true=y_true,
    y_pred=y_pred,
    y_proba=y_proba,
    inference_time=inference_time,
    num_samples=len(texts)
)
```

## Reproducibility

The benchmarking system ensures reproducibility through:
- **Fixed Random Seeds**: Consistent train/test splits
- **Template Mapping**: Saved numerical mappings
- **Configuration Files**: Version-controlled settings
- **Deterministic Processing**: Same preprocessing pipeline

## References

- **LogBERT**: Log Anomaly Detection via BERT
- **Drain**: An Online Log Parsing Approach  
- **LogHub**: A Large Collection of System Log Datasets
- **RoBERTa**: A Robustly Optimized BERT Pretraining Approach
