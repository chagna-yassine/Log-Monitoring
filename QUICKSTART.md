# Quick Start Guide

This guide will help you get started with benchmarking the log anomaly detection model.

## Prerequisites

- Python 3.8 or higher
- 10GB+ free disk space (for HDFS dataset)
- GPU recommended (but CPU works too)
- Internet connection (for downloading dataset and model)

## Installation

### 1. Clone or Set Up the Project

Navigate to the project directory:

```bash
cd 7030CEM
```

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate

# On Linux/Mac:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- PyTorch and Transformers (for the model)
- Pandas and NumPy (for data processing)
- Scikit-learn (for metrics)
- Other utilities

## Usage

### Option 1: Run Complete Pipeline (Easiest)

Run everything with one command:

```bash
python run_all.py
```

This will:
1. Download the HDFS dataset (~500MB compressed)
2. Preprocess the logs (may take 10-30 minutes)
3. Run benchmarking with the model
4. Save results to `results/` directory

### Option 2: Run Steps Individually

If you prefer to run steps separately:

#### Step 1: Download Dataset

```bash
python scripts/download_data.py
```

This downloads and extracts the HDFS dataset from Zenodo.

**Expected output:**
- `datasets/hdfs/HDFS.log` (~16GB raw log file)
- `datasets/hdfs/anomaly_label.csv` (ground truth labels)

#### Step 2: Preprocess Data

```bash
python scripts/preprocess.py
```

This runs the 6-step preprocessing pipeline:
1. Parse logs with Drain algorithm
2. Create template mapping
3. Build block-wise sequences
4. Assign anomaly labels
5. Convert to text format
6. Split into train/test sets

**Time:** 10-30 minutes depending on your system

**Expected output:**
- `datasets/hdfs/output/hdfs/` directory with processed files
- `hdfs_test.csv` - ready for benchmarking

#### Step 3: Run Benchmark

```bash
python scripts/benchmark.py
```

This:
- Loads the model from Hugging Face
- Runs inference on test set
- Computes comprehensive metrics
- Saves results

**Time:** 5-15 minutes depending on GPU/CPU

**Expected output:**
- `results/benchmark_results.json` - all metrics
- `results/predictions.csv` - individual predictions

### View Results

To view results in a readable format:

```bash
python scripts/view_results.py
```

Or open `results/benchmark_results.json` directly.

## Configuration

Edit `config.yaml` to customize:

```yaml
# Model settings
model:
  name: "Dumi2025/log-anomaly-detection-model-roberta"
  max_length: 512
  batch_size: 32    # Increase for GPU, decrease for CPU
  device: "cuda"    # Change to "cpu" if no GPU

# Preprocessing settings
preprocessing:
  train_ratio: 0.7
  random_seed: 42
```

## Expected Results

After successful completion, you should see:

```
results/
├── benchmark_results.json    # All metrics and statistics
└── predictions.csv           # Per-sample predictions
```

Key metrics include:
- **Accuracy**: Overall classification accuracy
- **Precision/Recall/F1**: For anomaly detection
- **AUC-ROC**: Area under ROC curve
- **Confusion Matrix**: True/False Positives/Negatives
- **Inference Time**: Performance metrics

## Troubleshooting

### Out of Memory Error

If you get OOM errors:
1. Reduce `batch_size` in `config.yaml`
2. Reduce `max_length` (e.g., to 256)
3. Use CPU instead of GPU

### Download Fails

If dataset download fails:
1. Check internet connection
2. Download manually from: https://zenodo.org/record/3227177/files/HDFS_1.tar.gz
3. Extract to `datasets/hdfs/` directory

### Model Download Fails

If model download fails:
1. Check Hugging Face is accessible
2. Check the model name is correct
3. Set `HF_HOME` environment variable for cache location

### Preprocessing Takes Too Long

The HDFS log file is large (~11M lines). On slower systems:
- First run may take 30+ minutes
- Progress updates every 100K lines
- You can test with a smaller sample first

## What's Next?

After successful benchmarking:

1. **Analyze Results**: Check `benchmark_results.json`
2. **Review Predictions**: Examine `predictions.csv` for errors
3. **Compare**: Compare with LogBERT baseline results
4. **Experiment**: Try different configurations in `config.yaml`
5. **Extend**: Add more datasets or models

## File Structure

```
7030CEM/
├── config.yaml              # Configuration
├── requirements.txt         # Dependencies
├── run_all.py              # Complete pipeline runner
├── QUICKSTART.md           # This file
├── README.md               # Full documentation
├── scripts/
│   ├── download_data.py    # Download dataset
│   ├── preprocess.py       # Preprocessing pipeline
│   ├── benchmark.py        # Run benchmark
│   └── view_results.py     # View results
├── src/
│   ├── parsers/            # Drain log parser
│   ├── preprocessing/      # Data preprocessing
│   ├── model/              # Model inference
│   └── evaluation/         # Metrics computation
├── datasets/               # Data storage (gitignored)
└── results/                # Results (gitignored)
```

## Support

For issues:
1. Check error messages carefully
2. Verify all files exist after each step
3. Check `config.yaml` settings
4. Ensure sufficient disk space and memory

## Citation

If you use this benchmarking system in your research:

```
@misc{log-anomaly-benchmark,
  title={Log Anomaly Detection Model Benchmarking System},
  year={2024},
  note={Benchmark for Dumi2025/log-anomaly-detection-model-roberta on HDFS dataset}
}
```

## References

- **Model**: Dumi2025/log-anomaly-detection-model-roberta (Hugging Face)
- **Dataset**: HDFS logs from LogHub (Zenodo)
- **Methodology**: Based on LogBERT preprocessing approach

