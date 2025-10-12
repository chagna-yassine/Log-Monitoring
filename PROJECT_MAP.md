# Project Map

## Complete File Structure

```
7030CEM/
│
├── 📋 Documentation (6 files)
│   ├── README.md                          Main documentation (comprehensive guide)
│   ├── QUICKSTART.md                      Quick start guide (getting started)
│   ├── PROJECT_OVERVIEW.md                Detailed architecture overview
│   ├── IMPLEMENTATION_SUMMARY.md          Implementation details
│   ├── PROJECT_MAP.md                     This file (navigation guide)
│   └── LICENSE                            MIT License
│
├── ⚙️ Configuration (2 files)
│   ├── config.yaml                        Main configuration file
│   └── requirements.txt                   Python dependencies
│
├── 🚀 Main Executables (2 files)
│   ├── run_all.py                         Run complete pipeline
│   └── example_usage.py                   Usage examples
│
├── 🐍 Source Code (src/)
│   ├── __init__.py                        Package initialization
│   │
│   ├── 📦 parsers/
│   │   ├── __init__.py
│   │   └── drain.py                       Drain log parser implementation
│   │
│   ├── 📦 preprocessing/
│   │   ├── __init__.py
│   │   ├── parser.py                      HDFS-specific log parser
│   │   ├── template_mapper.py             Event template mapping
│   │   ├── sequence_builder.py            Block-wise sequence builder
│   │   ├── data_splitter.py               Train/test data splitter
│   │   └── text_converter.py              Sequence to text converter
│   │
│   ├── 📦 model/
│   │   ├── __init__.py
│   │   └── inference.py                   Model loading and inference
│   │
│   └── 📦 evaluation/
│       ├── __init__.py
│       └── metrics.py                     Comprehensive metrics computation
│
├── 📜 Scripts (scripts/)
│   ├── __init__.py
│   ├── download_data.py                   Download HDFS dataset
│   ├── preprocess.py                      Run preprocessing pipeline
│   ├── benchmark.py                       Run model benchmarking
│   ├── view_results.py                    Display results
│   └── check_setup.py                     Verify project setup
│
├── 📁 Data Directory (datasets/)
│   ├── .gitkeep                           Git placeholder
│   └── hdfs/                              (Created by download_data.py)
│       ├── HDFS.log                       Raw HDFS logs (~16GB)
│       ├── anomaly_label.csv              Ground truth labels
│       └── output/hdfs/                   Processed files
│           ├── HDFS.log_structured.csv    Parsed logs
│           ├── HDFS.log_templates.csv     Event templates
│           ├── hdfs_log_templates.json    Template mapping
│           ├── hdfs_sequence.csv          Block sequences
│           ├── hdfs_sequence_labeled.csv  With labels
│           ├── hdfs_text.csv              Text format
│           ├── hdfs_train.csv             Training set
│           └── hdfs_test.csv              Test set
│
└── 📊 Results Directory (results/)
    ├── .gitkeep                           Git placeholder
    ├── benchmark_results.json             (Created by benchmark.py)
    └── predictions.csv                    Per-sample predictions

```

## File Purposes

### 📋 Documentation Files

| File | Purpose | When to Read |
|------|---------|--------------|
| `README.md` | Complete project documentation | First time setup |
| `QUICKSTART.md` | Step-by-step getting started | Want to start quickly |
| `PROJECT_OVERVIEW.md` | Architecture and design details | Understanding system design |
| `IMPLEMENTATION_SUMMARY.md` | Implementation details and stats | Understanding what was built |
| `PROJECT_MAP.md` | File structure navigation | Finding specific files |
| `LICENSE` | Legal license information | Understanding usage rights |

### ⚙️ Configuration Files

| File | Purpose | When to Edit |
|------|---------|--------------|
| `config.yaml` | Main configuration | Changing model/preprocessing settings |
| `requirements.txt` | Python dependencies | Adding new packages |

### 🚀 Main Executables

| File | Purpose | When to Run |
|------|---------|-------------|
| `run_all.py` | Complete pipeline | First time or full benchmark |
| `example_usage.py` | Usage examples | Learning how to use components |

### 🐍 Source Code Modules

#### parsers/
- `drain.py` - Implements Drain algorithm for log parsing
  - **Key Classes**: `Drain`, `LogCluster`, `Node`
  - **Main Function**: `parse()` - Parse logs and extract templates

#### preprocessing/
- `parser.py` - HDFS log parsing
  - **Key Class**: `HDFSLogParser`
  - **Main Function**: `parse_file()` - Parse entire log file
  
- `template_mapper.py` - Template ID mapping
  - **Key Class**: `TemplateMapper`
  - **Main Function**: `create_mapping()` - Create numerical mapping
  
- `sequence_builder.py` - Build log sequences
  - **Key Class**: `SequenceBuilder`
  - **Main Function**: `build_sequences()` - Create block sequences
  
- `data_splitter.py` - Split train/test
  - **Key Class**: `DataSplitter`
  - **Main Function**: `split_data()` - Split into train/test sets
  
- `text_converter.py` - Convert to text
  - **Key Class**: `TextConverter`
  - **Main Function**: `convert_sequence()` - Number to text

#### model/
- `inference.py` - Model inference
  - **Key Class**: `LogAnomalyDetector`
  - **Main Function**: `predict()` - Run inference

#### evaluation/
- `metrics.py` - Metrics computation
  - **Key Class**: `BenchmarkMetrics`
  - **Main Function**: `compute_all_metrics()` - Calculate all metrics

### 📜 Scripts

| Script | Purpose | Input | Output | Time |
|--------|---------|-------|--------|------|
| `download_data.py` | Download dataset | URL | HDFS logs | 5-10 min |
| `preprocess.py` | Preprocess data | Raw logs | Processed files | 10-30 min |
| `benchmark.py` | Run benchmark | Test set | Results JSON | 5-15 min |
| `view_results.py` | Display results | Results JSON | Formatted output | Instant |
| `check_setup.py` | Verify setup | Project files | Status report | Instant |

## Workflow Navigation

### For First Time Users

1. **Start Here**: `QUICKSTART.md`
2. **Then Read**: `README.md` (skim sections)
3. **Run**: `python scripts/check_setup.py`
4. **Execute**: `python run_all.py`
5. **View**: `python scripts/view_results.py`

### For Developers

1. **Architecture**: `PROJECT_OVERVIEW.md`
2. **Implementation**: `IMPLEMENTATION_SUMMARY.md`
3. **Code**: Browse `src/` directory
4. **Examples**: `example_usage.py`
5. **Extend**: Modify relevant modules

### For Researchers

1. **Methodology**: `README.md` → "Preprocessing Pipeline" section
2. **Run Benchmark**: `python run_all.py`
3. **Analyze**: `results/benchmark_results.json`
4. **Compare**: Check confusion matrix and metrics
5. **Investigate**: `results/predictions.csv` for error analysis

## Code Navigation Guide

### Want to understand...

**Log Parsing?**
- Read: `src/parsers/drain.py`
- See: `example_usage.py` → `example_parse_logs()`

**Data Processing?**
- Read: `src/preprocessing/` directory
- Start with: `parser.py` → `sequence_builder.py` → `text_converter.py`

**Model Inference?**
- Read: `src/model/inference.py`
- See: `example_usage.py` → `example_model_inference()`

**Evaluation Metrics?**
- Read: `src/evaluation/metrics.py`
- See: `example_usage.py` → `example_metrics_computation()`

**Complete Pipeline?**
- Read: `scripts/preprocess.py` → `scripts/benchmark.py`
- Or: `run_all.py`

## Quick Reference

### File Size Guide

| File Type | Typical Size | Location |
|-----------|--------------|----------|
| Python modules | 5-15 KB | `src/` |
| Scripts | 3-8 KB | `scripts/` |
| Documentation | 5-50 KB | Root directory |
| Raw HDFS logs | 16 GB | `datasets/hdfs/` |
| Processed data | 100-500 MB | `datasets/hdfs/output/` |
| Results | 10-150 MB | `results/` |

### Line Count Guide

| Module | Approx Lines | Complexity |
|--------|--------------|------------|
| `drain.py` | 300+ | High |
| `parser.py` | 200+ | Medium |
| `inference.py` | 150+ | Medium |
| `metrics.py` | 400+ | High |
| Other modules | 100-200 | Low-Medium |

## Import Paths

### For Scripts
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessing import HDFSLogParser
from model import LogAnomalyDetector
from evaluation import BenchmarkMetrics
```

### For External Use
```python
# Add project to path
import sys
sys.path.insert(0, '/path/to/7030CEM/src')

# Import modules
from parsers import Drain
from preprocessing import (
    HDFSLogParser,
    TemplateMapper,
    SequenceBuilder,
    DataSplitter,
    TextConverter
)
from model import LogAnomalyDetector
from evaluation import BenchmarkMetrics
```

## Dependency Tree

```
run_all.py
  └─> scripts/download_data.py
  └─> scripts/preprocess.py
        └─> src/parsers/drain.py
        └─> src/preprocessing/parser.py
        └─> src/preprocessing/template_mapper.py
        └─> src/preprocessing/sequence_builder.py
        └─> src/preprocessing/data_splitter.py
        └─> src/preprocessing/text_converter.py
  └─> scripts/benchmark.py
        └─> src/model/inference.py
        └─> src/evaluation/metrics.py
```

## Configuration Dependencies

```
config.yaml
  ├─> model (used by: scripts/benchmark.py)
  ├─> dataset (used by: scripts/download_data.py)
  ├─> output (used by: all scripts)
  ├─> preprocessing (used by: scripts/preprocess.py)
  └─> benchmark (used by: scripts/benchmark.py)
```

## Data Flow Diagram

```
[Raw Logs]
    ↓
[download_data.py]
    ↓
[HDFS.log + anomaly_label.csv]
    ↓
[preprocess.py]
    ├─> [drain.py] → Templates
    ├─> [parser.py] → Structured logs
    ├─> [template_mapper.py] → Mapping
    ├─> [sequence_builder.py] → Sequences
    └─> [data_splitter.py] → Train/Test
    ↓
[hdfs_test.csv]
    ↓
[benchmark.py]
    ├─> [inference.py] → Predictions
    └─> [metrics.py] → Results
    ↓
[benchmark_results.json]
    ↓
[view_results.py]
    ↓
[Formatted Output]
```

## Git Workflow

### Tracked Files
- All `.py` files
- All `.md` files
- `config.yaml`
- `requirements.txt`
- `LICENSE`
- `.gitkeep` files

### Ignored Files (.gitignore)
- `datasets/` (except `.gitkeep`)
- `results/` (except `.gitkeep`)
- `__pycache__/`
- `*.pyc`
- `.venv/`
- IDE files

## Search Index

**Find configuration**: → `config.yaml`  
**Find dependencies**: → `requirements.txt`  
**Find documentation**: → Root `*.md` files  
**Find log parser**: → `src/parsers/drain.py`  
**Find preprocessing**: → `src/preprocessing/`  
**Find model code**: → `src/model/inference.py`  
**Find metrics**: → `src/evaluation/metrics.py`  
**Find scripts**: → `scripts/`  
**Find examples**: → `example_usage.py`  
**Find data**: → `datasets/hdfs/`  
**Find results**: → `results/`  

---

**Last Updated**: October 2024  
**Total Files**: 32  
**Total Directories**: 8  
**Project Status**: ✅ Complete

