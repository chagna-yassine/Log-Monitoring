# Log Anomaly Detection Model Benchmarking

Comprehensive benchmarking system for evaluating the `Dumi2025/log-anomaly-detection-model-roberta` on HDFS logs.

## Overview

This project benchmarks a fine-tuned RoBERTa-base model for log anomaly detection using the HDFS dataset from LogHub. The implementation follows the LogBERT methodology for preprocessing and evaluation.

## Model Information

- **Model:** `Dumi2025/log-anomaly-detection-model-roberta`
- **Architecture:** RoBERTa-base (Transformer-based)
- **Task:** Binary sequence classification (Normal/Anomaly)
- **Source:** Hugging Face Model Hub

## Dataset

- **Dataset:** HDFS Logs (Hadoop Distributed File System)
- **Source:** LogHub via Zenodo
- **Size:** ~11 million log entries
- **Labels:** Block-level anomaly annotations
- **Anomaly Rate:** ~2-3% (realistic production distribution)

## Project Structure

```
7030CEM/
├── requirements.txt              # Python dependencies
├── config.yaml                   # Configuration file
├── README.md                     # This file
├── datasets/
│   └── hdfs/
│       ├── HDFS.log             # Raw log file
│       ├── anomaly_label.csv    # Ground truth labels
│       └── output/hdfs/
│           ├── HDFS.log_structured.csv
│           ├── HDFS.log_templates.csv
│           ├── hdfs_sequence.csv
│           └── hdfs_log_templates.json
├── src/
│   ├── parsers/
│   │   └── drain.py             # Drain log parser
│   ├── preprocessing/
│   │   ├── parser.py            # Log parsing pipeline
│   │   ├── template_mapper.py  # Template mapping
│   │   ├── sequence_builder.py # Block-wise sequences
│   │   └── data_splitter.py    # Train/test split
│   ├── model/
│   │   └── inference.py         # Model loading & inference
│   └── evaluation/
│       └── metrics.py           # Benchmark metrics
├── scripts/
│   ├── download_data.py         # Download HDFS dataset
│   ├── preprocess.py            # Run preprocessing pipeline
│   └── benchmark.py             # Run benchmarking
└── results/
    └── benchmark_results.json   # Final results

```

## Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Download Dataset

```bash
python scripts/download_data.py
```

### 2. Preprocess Data

```bash
python scripts/preprocess.py
```

This runs the complete preprocessing pipeline:
- Log parsing using Drain algorithm
- Template mapping and numbering
- Block-wise sequence creation
- Label assignment
- Text conversion for model input
- Train/test split (70/30 for normal blocks)

### 3. Run Benchmark

```bash
python scripts/benchmark.py
```

This will:
- Load the model from Hugging Face
- Run inference on the test set
- Compute comprehensive metrics
- Save results to `results/benchmark_results.json`

### 4. Run Complete Pipeline

```bash
# Run all steps at once
python scripts/download_data.py && python scripts/preprocess.py && python scripts/benchmark.py
```

## Preprocessing Pipeline

### Step 1: Log Parsing
- Parses raw HDFS logs using Drain algorithm
- Extracts structured information (timestamp, level, component, content)
- Generates event templates using regex patterns

### Step 2: Template Mapping
- Creates numerical IDs for each unique event template
- Sorts by frequency and assigns sequential numbers
- Saves mapping for reproducibility

### Step 3: Block-wise Sequences
- Groups log events by HDFS block ID
- Creates sequences of event IDs per block
- Handles block ID extraction and grouping

### Step 4: Label Assignment
- Loads ground truth from `anomaly_label.csv`
- Maps block IDs to Normal (0) or Anomaly (1)
- Validates label coverage

### Step 5: Text Conversion
- Converts numerical event sequences to text
- Maps event IDs back to readable templates
- Formats with separators for tokenization

### Step 6: Train/Test Split
- Follows LogBERT methodology
- 70% normal blocks for training reference
- 30% normal + all anomalous blocks for testing
- Maintains realistic anomaly distribution

## Benchmark Metrics

The system computes:
- **Accuracy:** Overall classification accuracy
- **Precision, Recall, F1:** Binary, weighted, macro, and micro averages
- **AUC-ROC:** Area under the ROC curve
- **Confusion Matrix:** True/False Positives/Negatives
- **Per-class Performance:** Normal vs Anomaly metrics
- **Inference Time:** Throughput and latency measurements

## Configuration

Edit `config.yaml` to customize:
- Model parameters (batch size, max length)
- Dataset paths and URLs
- Preprocessing parameters (Drain settings, regex patterns)
- Evaluation metrics
- Train/test split ratios

## Results

Results are saved to `results/benchmark_results.json` with:
- All computed metrics
- Confusion matrix
- Per-class statistics
- Inference performance
- Dataset statistics
- Timestamp and configuration used

## References

- LogBERT: Log Anomaly Detection via BERT
- Drain: An Online Log Parsing Approach
- LogHub: A Large Collection of System Log Datasets

## License

This benchmarking system is for research and educational purposes.

