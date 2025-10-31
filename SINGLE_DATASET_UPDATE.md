# Script Update: Single Dataset Analysis

## ✅ Changes Made

The analysis script has been updated to analyze **ONE dataset at a time** instead of all 8 datasets simultaneously.

### Why the Change?

Loading all 8 AIT datasets at once causes memory issues, especially in Google Colab. The new version is:
- ✅ **Memory efficient** - Only loads one dataset
- ✅ **Faster** - No need to wait for all datasets
- ✅ **Flexible** - Choose which dataset to analyze

## New Usage

### Command Line

```bash
# Analyze a specific dataset
python scripts/analyze_all_ait_datasets.py --repo chYassine/ait-fox-raw-v02

# With token prompt (for private datasets)
python scripts/analyze_all_ait_datasets.py --repo chYassine/ait-fox-raw-v02 --token

# List available datasets
python scripts/analyze_all_ait_datasets.py --list

# Interactive mode (prompts for dataset)
python scripts/analyze_all_ait_datasets.py
```

### In Python/Colab

```python
from analyze_all_ait_datasets import analyze_dataset

# Analyze one dataset
analyze_dataset('chYassine/ait-fox-raw-v02')
```

### With Authentication

```python
# Method 1: Use token prompt
!python scripts/analyze_all_ait_datasets.py --repo chYassine/ait-fox-raw-v02 --token
# Then enter token when prompted

# Method 2: Login first
from huggingface_hub import login
login(token="YOUR_TOKEN")

from analyze_all_ait_datasets import analyze_dataset
analyze_dataset('chYassine/ait-fox-raw-v02')
```

## Key Features

### 1. Single Dataset Analysis
- Loads only the dataset you specify
- Avoids memory issues
- Faster analysis

### 2. Token Input
- Interactive prompt for HuggingFace token
- Use `--token` flag to enable prompt
- Secure password-style input (hidden)

### 3. Interactive Mode
- Run without arguments
- Select from list of 8 datasets
- Enter number (1-8) or full repo name

### 4. Colab Compatible
- Handles Jupyter kernel arguments
- Falls back gracefully if argparse fails
- Works in notebooks and terminals

## Output Changes

### File Naming
Output files now use the dataset name:

**Old**: `ait_datasets_analysis_summary.json`  
**New**: `{dataset_name}_analysis_summary.json`

Examples:
- `ait-fox-raw-v02_analysis_summary.json`
- `ait-wilson-raw-v01_analysis_summary.json`

### Statistics
- Focus on the single dataset
- No longer shows "Total Datasets: 8"
- Shows dataset name prominently

## Analyzing Multiple Datasets

If you want to analyze all 8 datasets, run them sequentially:

### Command Line
```bash
python scripts/analyze_all_ait_datasets.py --repo chYassine/ait-wilson-raw-v01
python scripts/analyze_all_ait_datasets.py --repo chYassine/ait-wheeler-raw-v01
python scripts/analyze_all_ait_datasets.py --repo chYassine/ait-wardbeck-raw-v01
# ... etc
```

### Python Loop
```python
from analyze_all_ait_datasets import analyze_dataset

datasets = [
    'chYassine/ait-wilson-raw-v01',
    'chYassine/ait-wheeler-raw-v01',
    'chYassine/ait-wardbeck-raw-v01',
    'chYassine/ait-shaw-raw-v01',
    'chYassine/ait-santos-raw-v01',
    'chYassine/air-russellmitchell-raw-v01',
    'chYassine/ait-harrison-raw-v01',
    'chYassine/ait-fox-raw-v02'
]

for dataset in datasets:
    print(f"\n{'='*80}")
    print(f"Analyzing: {dataset}")
    print('='*80)
    analyze_dataset(dataset)
```

Each dataset will be:
- Analyzed separately
- Saved to its own JSON file
- Memory cleared between runs

## Available Datasets

1. `chYassine/ait-wilson-raw-v01`
2. `chYassine/ait-wheeler-raw-v01`
3. `chYassine/ait-wardbeck-raw-v01`
4. `chYassine/ait-shaw-raw-v01`
5. `chYassine/ait-santos-raw-v01`
6. `chYassine/air-russellmitchell-raw-v01`
7. `chYassine/ait-harrison-raw-v01`
8. `chYassine/ait-fox-raw-v02`

## Migration Guide

### Old Way (No Longer Works)
```python
# This will cause errors now
from analyze_all_ait_datasets import analyze_datasets
analyze_datasets([...])  # Function no longer exists
```

### New Way
```python
# Option 1: Single dataset
from analyze_all_ait_datasets import analyze_dataset
analyze_dataset('chYassine/ait-fox-raw-v02')

# Option 2: Loop through multiple
for repo in repos:
    analyze_dataset(repo)
```

## Benefits

✅ **Memory Efficient** - No more OOM errors  
✅ **Faster Start** - Begin analysis immediately  
✅ **Better Control** - Choose what to analyze  
✅ **Clear Output** - One dataset, one file  
✅ **Colab Ready** - Works perfectly in notebooks  
✅ **Secure Auth** - Password-style token input  

## Updated Documentation

- `QUICK_ANALYSIS_GUIDE.md` - Updated examples
- `COLAB_USAGE.md` - New Colab instructions
- `SINGLE_DATASET_UPDATE.md` - This file

## Example Session

```bash
$ python scripts/analyze_all_ait_datasets.py

📊 Available Datasets:
  1. chYassine/ait-wilson-raw-v01
  2. chYassine/ait-wheeler-raw-v01
  3. chYassine/ait-wardbeck-raw-v01
  4. chYassine/ait-shaw-raw-v01
  5. chYassine/ait-santos-raw-v01
  6. chYassine/air-russellmitchell-raw-v01
  7. chYassine/ait-harrison-raw-v01
  8. chYassine/ait-fox-raw-v02

Enter dataset number or full repository name: 8

🚀 Starting analysis of: chYassine/ait-fox-raw-v02
Loading: chYassine/ait-fox-raw-v02
✅ Successfully loaded: 1,234,567 entries

[Analysis output...]

✅ Summary saved to: ait-fox-raw-v02_analysis_summary.json
```

---

**Updated**: January 2024  
**Version**: 2.0.0  
**Status**: Production Ready ✅

