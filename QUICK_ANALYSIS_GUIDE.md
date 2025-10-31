# Quick Start: Analyze AIT Datasets

## ⚠️ Updated for Single Dataset Analysis

To avoid memory issues, the script now analyzes ONE dataset at a time.

## One-Command Analysis

Analyze a specific dataset:

```bash
python scripts/analyze_all_ait_datasets.py --repo chYassine/ait-fox-raw-v02
```

With HuggingFace token (for private datasets):

```bash
python scripts/analyze_all_ait_datasets.py --repo chYassine/ait-fox-raw-v02 --token
```

**Note**: Works perfectly in Google Colab! See `COLAB_USAGE.md` for Colab-specific instructions.

## What You'll Get

### 📊 Comprehensive Statistics
- ✅ Total log entries in the dataset
- ✅ Top hosts by log count
- ✅ Log type distribution
- ✅ Text vs binary breakdown
- ✅ Content length statistics

### 📋 Sample Logs
- ✅ First example from each log type
- ✅ First example from top 10 hosts
- ✅ 5 diverse random samples

### 💾 Saved Summary
- ✅ JSON file: `{dataset_name}_analysis_summary.json`

## Available Datasets

Choose one to analyze:

1. `chYassine/ait-wilson-raw-v01`
2. `chYassine/ait-wheeler-raw-v01`
3. `chYassine/ait-wardbeck-raw-v01`
4. `chYassine/ait-shaw-raw-v01`
5. `chYassine/ait-santos-raw-v01`
6. `chYassine/air-russellmitchell-raw-v01`
7. `chYassine/ait-harrison-raw-v01`
8. `chYassine/ait-fox-raw-v02`

List them with:
```bash
python scripts/analyze_all_ait_datasets.py --list
```

## Installation

```bash
pip install datasets huggingface_hub pandas
```

## Need Authentication?

If datasets are private, set up your Hugging Face token:

```bash
python scripts/setup_hf_token.py
```

## Full Documentation

See `ANALYZE_ALL_DATASETS_README.md` for complete details.

## Example Output

```
================================================================================
  AIT DATASET ANALYSIS
================================================================================
Dataset: chYassine/ait-fox-raw-v02

Total Log Entries: 1,234,567
Unique Hosts: 15
Unique Log Types: 8

Top 15 Hosts by Log Count:
  host1: 345,678 (28.0%)
  host2: 234,567 (19.0%)
  ...

Log Types Distribution:
  apache_access: 678,432 (54.9%)
  apache_error: 234,567 (19.0%)
  ...

📋 Sample Logs:
[Multiple samples from different types and hosts]

✅ Summary saved to: ait-fox-raw-v02_analysis_summary.json
```

## Interactive Mode

If you run without arguments, the script prompts you:

```bash
python scripts/analyze_all_ait_datasets.py
```

Then enter the dataset number or full name when prompted.

## That's It!

Just run the script and explore your data! 🚀

