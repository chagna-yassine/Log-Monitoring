# BGL Dataset Quick Start Guide

This guide will help you benchmark the log anomaly detection model on the BGL (Blue Gene/L) dataset.

## Prerequisites

- Python 3.8 or higher
- 5GB+ free disk space (for BGL dataset)
- GPU recommended (but CPU works too)
- Internet connection (for downloading dataset and model)

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure for BGL

Edit `config.yaml` and set:

```yaml
# Change this line
dataset_name: "BGL"  # Instead of "HDFS"
```

## Usage

### Option 1: Run Complete BGL Pipeline (Easiest)

```bash
python run_all.py
```

This will automatically:
1. Download the BGL dataset (~500MB compressed)
2. Preprocess the logs (may take 10-30 minutes)
3. Run benchmarking with the model
4. Save results to `results/bgl_benchmark_results.json`

### Option 2: Run BGL Steps Individually

#### Step 1: Download BGL Dataset

```bash
python scripts/download_data_bgl.py
```

**Expected output:**
- `datasets/bgl/BGL_2k.log` (~2GB raw log file)
- `datasets/bgl/BGL_2k.log_structured.csv` (ground truth labels)

#### Step 2: Preprocess BGL Data

```bash
python scripts/preprocess_bgl.py
```

This runs the 6-step preprocessing pipeline:
1. Parse logs with Drain algorithm (adapted for BGL format)
2. Create template mapping
3. Build session-wise sequences (BGL doesn't have block IDs)
4. Assign anomaly labels
5. Convert to text format
6. Split into train/test sets

**Time:** 10-30 minutes depending on your system

**Expected output:**
- `datasets/bgl/output/bgl/` directory with processed files
- `bgl_test.csv` - ready for benchmarking

#### Step 3: Run BGL Benchmark

```bash
python scripts/benchmark_bgl.py
```

This:
- Loads the model from Hugging Face
- Runs inference on BGL test set
- Computes comprehensive metrics
- Saves results

**Time:** 5-15 minutes depending on GPU/CPU

**Expected output:**
- `results/bgl_benchmark_results.json` - all metrics
- `results/bgl_predictions.csv` - individual predictions

### Option 3: Use Unified Runner

```bash
python scripts/run_dataset.py
```

This reads the dataset configuration from `config.yaml` and runs the appropriate pipeline.

## View Results

### BGL Results Only

```bash
python scripts/view_results.py results/bgl_benchmark_results.json
```

### Compare HDFS vs BGL Results

```bash
python scripts/compare_results.py
```

This compares:
- `results/benchmark_results.json` (HDFS)
- `results/bgl_benchmark_results.json` (BGL)

## BGL Dataset Information

### Dataset Characteristics
- **Source**: Blue Gene/L supercomputer logs
- **Size**: ~4.7 million log entries
- **Time Period**: 7 months of production logs
- **Labels**: ~348,460 labeled anomalies
- **Anomaly Rate**: ~7.4% (higher than HDFS)

### BGL Log Format
BGL logs have multiple possible formats:
- `<Timestamp> <NodeID> <Level> <Component>: <Content>`
- `<Timestamp> <Level> <Component> <NodeID>: <Content>`
- `<NodeID> <Timestamp> <Level> <Component>: <Content>`

The parser automatically detects and handles different formats.

### Key Differences from HDFS

| Aspect | HDFS | BGL |
|--------|------|-----|
| **Grouping** | Block IDs | Session-based (NodeID + time) |
| **Log Format** | Single format | Multiple formats |
| **Anomaly Rate** | ~2-3% | ~7.4% |
| **Sequence Length** | Variable | Variable |
| **Node Types** | DataNodes | Compute nodes |

## Configuration

### BGL-Specific Settings in `config.yaml`

```yaml
# Dataset selection
dataset_name: "BGL"

# BGL dataset configuration
bgl_dataset:
  name: "BGL"
  url: "https://zenodo.org/record/3227177/files/BGL_2k.tar.gz"
  base_path: "datasets/bgl"
  raw_log: "BGL_2k.log"
  label_file: "BGL_2k.log_structured.csv"

# BGL output paths
output:
  bgl:
    base_path: "datasets/bgl/output/bgl"
    structured_log: "BGL.log_structured.csv"
    templates: "BGL.log_templates.csv"
    sequences: "bgl_sequence.csv"
    template_mapping: "bgl_log_templates.json"
```

## Expected Results

### After Preprocessing
```
datasets/bgl/output/bgl/
├── BGL.log_structured.csv      (~800MB)
├── BGL.log_templates.csv       (~50KB)
├── bgl_log_templates.json      (~100KB)
├── bgl_sequence.csv           (~200MB)
├── bgl_sequence_labeled.csv   (~220MB)
├── bgl_text.csv               (~400MB)
├── bgl_train.csv              (~280MB)
└── bgl_test.csv               (~120MB)
```

### After Benchmarking
```
results/
├── bgl_benchmark_results.json  (~10KB)
└── bgl_predictions.csv        (~120MB)
```

## Key Metrics

The BGL benchmark computes the same comprehensive metrics as HDFS:
- Accuracy, Precision, Recall, F1-Score
- Binary, Weighted, Macro, Micro averages
- AUC-ROC, Confusion Matrix
- Per-class statistics (Normal vs Anomaly)
- Inference time and throughput

## Troubleshooting

### BGL-Specific Issues

1. **Parser Fails on Some Logs**
   - The BGL parser uses multiple patterns and fallback parsing
   - Some logs may not parse perfectly - this is normal
   - Check the parsing statistics in the output

2. **Session Grouping Issues**
   - BGL uses session-based grouping instead of block IDs
   - Sessions are created based on NodeID + time windows
   - Adjust session window in `BGLSequenceBuilder` if needed

3. **Label Assignment Problems**
   - BGL labels may be in different formats
   - The parser handles multiple label formats automatically
   - Check label distribution in the output

### General Issues

1. **Out of Memory**
   - Reduce `batch_size` in `config.yaml`
   - Reduce `max_length` to 256

2. **Download Fails**
   - Check internet connection
   - Verify Zenodo URL is accessible

3. **Model Download Fails**
   - Check Hugging Face access
   - Verify model name is correct

## Performance Expectations

### BGL vs HDFS Comparison

| Metric | HDFS | BGL (Expected) |
|--------|------|----------------|
| **Dataset Size** | ~11M entries | ~4.7M entries |
| **Test Samples** | ~16K | ~10-15K |
| **Anomaly Rate** | ~2-3% | ~7.4% |
| **Processing Time** | 10-30 min | 10-30 min |
| **Inference Time** | 5-15 min | 5-15 min |

### Expected Performance

Based on typical BGL results:
- **Accuracy**: 85-95%
- **Precision**: 80-90%
- **Recall**: 70-85%
- **F1-Score**: 75-85%
- **AUC-ROC**: 0.85-0.95

## Next Steps

After BGL benchmarking:

1. **Analyze Results**: Check `results/bgl_benchmark_results.json`
2. **Compare with HDFS**: Run `python scripts/compare_results.py`
3. **Review Predictions**: Examine `results/bgl_predictions.csv`
4. **Experiment**: Try different configurations
5. **Research**: Compare with BGL baseline results from literature

## File Structure for BGL

```
7030CEM/
├── config.yaml                    # Set dataset_name: "BGL"
├── scripts/
│   ├── download_data_bgl.py       # Download BGL dataset
│   ├── preprocess_bgl.py          # BGL preprocessing
│   ├── benchmark_bgl.py           # BGL benchmarking
│   └── compare_results.py         # Compare HDFS vs BGL
├── src/preprocessing/
│   ├── bgl_parser.py              # BGL-specific parser
│   └── bgl_sequence_builder.py    # BGL sequence builder
├── datasets/bgl/                  # BGL data storage
└── results/                       # Results storage
```

## Support

For BGL-specific issues:
1. Check this `BGL_QUICKSTART.md`
2. Review preprocessing output for parsing statistics
3. Compare with HDFS results for validation
4. Check configuration in `config.yaml`

---

**Ready to benchmark BGL!** 🚀

Simply set `dataset_name: "BGL"` in `config.yaml` and run:
```bash
python run_all.py
```
