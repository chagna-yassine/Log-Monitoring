# Project Overview: Log Anomaly Detection Benchmarking

## Executive Summary

This project implements a complete benchmarking system for evaluating the **Dumi2025/log-anomaly-detection-model-roberta** model on HDFS (Hadoop Distributed File System) logs. The system follows the LogBERT methodology for preprocessing and provides comprehensive evaluation metrics.

## Project Goals

1. **Benchmark a state-of-the-art model** for log anomaly detection
2. **Use authentic production data** from HDFS systems (~11M log entries)
3. **Follow established methodology** (LogBERT preprocessing pipeline)
4. **Provide comprehensive metrics** for thorough evaluation
5. **Create reusable framework** for future model comparisons

## Architecture Overview

### Data Flow

```
Raw HDFS Logs (HDFS.log)
          ↓
    [Drain Parser]
          ↓
Structured Logs + Templates
          ↓
  [Template Mapper]
          ↓
   Numerical Mapping
          ↓
 [Sequence Builder]
          ↓
Block-wise Sequences + Labels
          ↓
  [Text Converter]
          ↓
   Text Sequences
          ↓
[Train/Test Split]
          ↓
Test Set (hdfs_test.csv)
          ↓
  [Model Inference]
          ↓
   Predictions
          ↓
  [Metrics Computation]
          ↓
Benchmark Results
```

## Component Details

### 1. Drain Log Parser (`src/parsers/drain.py`)

**Purpose**: Extract event templates from raw logs

**Algorithm**: Drain (Fixed Depth Tree-based online log parsing)

**Key Features**:
- Parse tree with configurable depth
- Similarity-based clustering
- Regex-based variable masking
- Incremental template extraction

**Configuration**:
- `depth`: 4 (parse tree depth)
- `st`: 0.4 (similarity threshold)
- Regex patterns for block IDs, IPs, file paths

### 2. Preprocessing Pipeline (`src/preprocessing/`)

#### 2.1 HDFSLogParser (`parser.py`)
- Parses HDFS log format: `<Date> <Time> <Pid> <Level> <Component>: <Content>`
- Extracts structured fields
- Assigns event IDs using Drain

#### 2.2 TemplateMapper (`template_mapper.py`)
- Creates numerical mapping for templates
- Sorts by frequency
- Assigns sequential IDs (1, 2, 3, ...)
- Saves mapping as JSON for reproducibility

#### 2.3 SequenceBuilder (`sequence_builder.py`)
- Extracts block IDs from log content
- Groups events by block ID
- Creates sequences of event numbers
- Adds anomaly labels from ground truth

#### 2.4 DataSplitter (`data_splitter.py`)
- Splits normal blocks: 70% train, 30% test
- All anomalous blocks go to test set
- Maintains realistic anomaly distribution (~2-3%)

#### 2.5 TextConverter (`text_converter.py`)
- Converts numerical sequences to text
- Maps event numbers back to templates
- Formats with separator for tokenization

### 3. Model Inference (`src/model/inference.py`)

**Model**: RoBERTa-base fine-tuned for sequence classification

**Features**:
- Automatic device selection (CUDA/CPU)
- Batch processing for efficiency
- Progress tracking
- Returns predictions and probabilities

**Configuration**:
- `max_length`: 512 tokens
- `batch_size`: 32 (adjustable)
- Output: Binary classification (Normal/Anomaly)

### 4. Evaluation Metrics (`src/evaluation/metrics.py`)

**Comprehensive Metrics**:
- **Accuracy**: Overall classification accuracy
- **Binary Metrics**: Precision, Recall, F1 (for anomaly class)
- **Weighted Metrics**: Account for class imbalance
- **Macro Metrics**: Treat all classes equally
- **Micro Metrics**: Global aggregation
- **AUC-ROC**: Area under ROC curve
- **Confusion Matrix**: TP, TN, FP, FN
- **Per-class Statistics**: Normal vs Anomaly
- **Performance**: Throughput, latency

## Orchestration Scripts

### `scripts/download_data.py`
Downloads and extracts HDFS dataset from Zenodo

**Output**: 
- `datasets/hdfs/HDFS.log` (~16GB)
- `datasets/hdfs/anomaly_label.csv`

### `scripts/preprocess.py`
Runs complete 6-step preprocessing pipeline

**Time**: 10-30 minutes

**Output**:
- Structured logs
- Event templates
- Template mapping
- Block sequences
- Train/test splits

### `scripts/benchmark.py`
Loads model and runs inference on test set

**Time**: 5-15 minutes

**Output**:
- `results/benchmark_results.json`
- `results/predictions.csv`

### `scripts/view_results.py`
Displays benchmark results in readable format

### `scripts/check_setup.py`
Verifies project setup and dependencies

### `run_all.py`
Runs complete pipeline end-to-end

## Dataset: HDFS Logs

**Source**: LogHub repository via Zenodo

**Statistics**:
- ~11 million raw log entries
- ~575,000 unique HDFS blocks
- ~16,000 test sequences
- Anomaly rate: ~2-3%
- Time period: Production system logs

**Log Format**:
```
081109 203615 148 INFO dfs.DataNode$PacketResponder: Received block blk_38 from /10.250.19.102
```

**Anomaly Types**:
- Block corruption
- Network failures
- Node failures
- Timeout errors

## Evaluation Methodology

### Following LogBERT Approach

1. **Same preprocessing**: Drain parser with identical parameters
2. **Same data split**: 70/30 train/test for normal blocks
3. **Same input format**: Text sequences of event templates
4. **Same metrics**: Precision, Recall, F1, etc.

### Fair Comparison

- Test set has realistic anomaly distribution
- No data leakage between train/test
- Temporal consistency maintained
- All anomalous blocks in test set only

## Results Interpretation

### Key Metrics to Focus On

1. **F1-Score (Anomaly)**: Primary metric for imbalanced data
2. **Recall (Anomaly)**: How many actual anomalies detected
3. **Precision (Anomaly)**: How many predictions are correct
4. **AUC-ROC**: Overall discrimination ability

### Success Criteria

Based on LogBERT results, good performance would be:
- F1-Score > 0.90
- Precision > 0.95
- Recall > 0.85
- AUC-ROC > 0.95

## Technical Considerations

### Memory Requirements

- **Download**: 500MB compressed, 16GB extracted
- **Preprocessing**: ~2GB RAM
- **Inference**: 4-8GB RAM (CPU), 2-4GB VRAM (GPU)

### Computational Requirements

- **CPU**: Multi-core recommended
- **GPU**: Optional but significantly faster
- **Time**: 
  - First run: 20-50 minutes total
  - Inference only: 5-15 minutes

### Optimization Tips

1. **Reduce batch_size** if OOM errors
2. **Reduce max_length** to 256 for faster processing
3. **Use GPU** for 5-10x speedup
4. **Sample data** for quick testing

## Extensibility

### Adding New Datasets

1. Implement dataset-specific parser
2. Adapt log format regex
3. Ensure label format compatibility
4. Update configuration

### Adding New Models

1. Implement model wrapper in `src/model/`
2. Ensure compatible input/output format
3. Update configuration with model name
4. Run benchmarking

### Custom Metrics

1. Add metric functions to `src/evaluation/metrics.py`
2. Update `compute_all_metrics()` method
3. Update result display in `view_results.py`

## File Outputs

### Preprocessing Outputs

```
datasets/hdfs/output/hdfs/
├── HDFS.log_structured.csv      # Parsed logs with event IDs
├── HDFS.log_templates.csv       # Event templates + frequencies
├── hdfs_log_templates.json      # Template mapping
├── hdfs_sequence.csv            # Block sequences (numbers)
├── hdfs_sequence_labeled.csv    # With labels
├── hdfs_text.csv                # Text sequences
├── hdfs_train.csv               # Training set
└── hdfs_test.csv                # Test set
```

### Benchmark Outputs

```
results/
├── benchmark_results.json       # All metrics + metadata
└── predictions.csv              # Per-sample predictions
```

## Configuration Options

### `config.yaml` Sections

```yaml
model:
  name: "Dumi2025/log-anomaly-detection-model-roberta"
  max_length: 512
  batch_size: 32
  device: "cuda"  # or "cpu"

dataset:
  name: "HDFS"
  url: "https://zenodo.org/record/3227177/files/HDFS_1.tar.gz"
  base_path: "datasets/hdfs"

preprocessing:
  drain:
    depth: 4
    st: 0.4
    rex: [...]
  train_ratio: 0.7
  random_seed: 42

benchmark:
  metrics: ["accuracy", "precision", "recall", "f1", "auc_roc"]
```

## Best Practices

1. **Always run preprocessing before benchmarking**
2. **Keep original data intact** (outputs go to separate directory)
3. **Use consistent random seed** for reproducibility
4. **Save all results** with timestamps
5. **Compare multiple runs** for stability
6. **Document any configuration changes**

## Known Limitations

1. **Large dataset**: Initial download and processing is time-consuming
2. **Memory intensive**: Requires sufficient RAM/VRAM
3. **Single model**: Currently supports one model format
4. **Binary classification**: Only Normal/Anomaly (no multi-class)

## Future Enhancements

1. Support for multiple datasets (BGL, Thunderbird, etc.)
2. Multi-model comparison framework
3. Hyperparameter tuning utilities
4. Visualization tools (ROC curves, confusion matrices)
5. Real-time inference capabilities
6. Explainability features (attention visualization)

## Research Context

This benchmarking system enables:
- Model performance evaluation
- Comparison with baselines (LogBERT, DeepLog, etc.)
- Ablation studies
- Transfer learning experiments
- Production deployment feasibility assessment

## References

1. **LogBERT**: "LogBERT: Log Anomaly Detection via BERT" (2021)
2. **Drain**: "Drain: An Online Log Parsing Approach with Fixed Depth Tree" (2017)
3. **HDFS Dataset**: LogHub repository (Zenodo)
4. **RoBERTa**: "RoBERTa: A Robustly Optimized BERT Pretraining Approach" (2019)

## Contact & Support

For issues or questions:
1. Check `QUICKSTART.md` for common problems
2. Review `README.md` for detailed documentation
3. Run `python scripts/check_setup.py` to verify setup
4. Check configuration in `config.yaml`

---

**Last Updated**: October 2024
**Version**: 1.0.0
**Status**: Production Ready

