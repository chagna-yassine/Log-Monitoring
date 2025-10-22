# AIT-LDS Log Anomaly Detection Processing Pipeline

This repository contains a complete processing pipeline for the AIT-LDS (Austrian Institute of Technology Log Data Sets) for log anomaly detection benchmarking.

## Overview

The AIT-LDS dataset is a comprehensive collection of synthetic log data suitable for evaluation of intrusion detection systems. This pipeline processes the multi-host log structure and prepares it for machine learning models.

## Dataset Information

- **Source**: [Zenodo Record 5789064](https://zenodo.org/records/5789064)
- **License**: CC-BY-NC-SA-4.0 (Non-commercial use only)
- **Size**: 14-39 GB per dataset
- **Datasets**: 8 testbeds (fox, harrison, russellmitchell, santos, shaw, wardbeck, wheeler, wilson)

## Features

- ✅ Multi-host log parsing
- ✅ Attack label integration
- ✅ Drain-based template extraction
- ✅ Sequence building with session grouping
- ✅ Text conversion for LLM input
- ✅ Train/test splitting
- ✅ Complete benchmarking pipeline

## Quick Start

### 1. Download Dataset
```bash
python scripts/download_data_ait.py
```

### 2. Process Data
```bash
python scripts/preprocess_ait.py
```

### 3. Run Benchmark
```bash
python scripts/benchmark_ait.py
```

### 4. Complete Pipeline
```bash
python run_all.py
```

## Configuration

Edit `config.yaml` to select different datasets:

```yaml
ait_dataset:
  selected_dataset: "fox"  # Change to any available dataset
```

Available datasets:
- fox (26 GB, High scan volume)
- harrison (27 GB, High scan volume)  
- russellmitchell (14 GB, Low scan volume)
- santos (17 GB, Low scan volume)
- shaw (27 GB, Low scan volume)
- wardbeck (26 GB, Low scan volume)
- wheeler (30 GB, High scan volume)
- wilson (39 GB, High scan volume)

## Log Types Supported

- Apache access/error logs
- Authentication logs
- DNS logs
- VPN logs
- Audit logs
- Suricata logs
- System logs

## Attack Types Detected

- Scans (nmap, WPScan, dirb)
- Webshell upload (CVE-2020-24186)
- Password cracking (John the Ripper)
- Privilege escalation
- Remote command execution
- Data exfiltration (DNSteal)

## Output Files

The pipeline generates:
- `ait_structured.csv` - Parsed logs with templates
- `ait_sequence.csv` - Event sequences
- `ait_text.csv` - Text sequences for model input
- `ait_train.csv` / `ait_test.csv` - Train/test splits
- `ait_log_templates.json` - Template mapping

## Citation

If you use this processing pipeline, please cite:

```bibtex
@article{landauer2022maintainable,
  title={Maintainable Log Datasets for Evaluation of Intrusion Detection Systems},
  author={Landauer, Max and Skopik, Florian and Frank, Maximilian and Hotwagner, Wolfgang and Wurzenberger, Markus and Rauber, Andreas},
  journal={IEEE Transactions on Dependable and Secure Computing},
  volume={20},
  number={4},
  pages={3466--3482},
  year={2022},
  publisher={IEEE}
}
```

## Requirements

- Python 3.8+
- 50+ GB free disk space
- Internet connection for dataset download

## License

This processing pipeline is provided under the same license as the original AIT-LDS dataset (CC-BY-NC-SA-4.0). Please respect the licensing terms and use only for non-commercial purposes.
