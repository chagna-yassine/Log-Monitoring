# Google Colab Log Cleaning Guide

This guide will walk you through running the complete log cleaning pipeline in Google Colab.

## 📋 Prerequisites

- Google account with access to Colab
- HuggingFace dataset: `chYassine/ait-fox-raw-v02` (or your dataset)
- (Optional) HuggingFace account and token to push cleaned dataset

## 🚀 Quick Start (3 Steps)

### Step 1: Run Discovery Script

This analyzes your raw logs to understand host/log_type distribution.

**Open a new Colab notebook and run:**

```python
# Install dependencies
!pip -q install datasets pandas tqdm

import os
import json
import csv
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

HF_REPO = "chYassine/ait-fox-raw-v02"  # Your repo
OUT_DIR = "artifacts/cleaning/discovery"
os.makedirs(OUT_DIR, exist_ok=True)

# ... [Copy entire discovery cell from earlier, or use the script provided]
```

**Expected output:**
- `artifacts/cleaning/discovery/hosts.csv`
- `artifacts/cleaning/discovery/log_types.csv`
- `artifacts/cleaning/discovery/host_logtype_crosstab.csv`
- `artifacts/cleaning/discovery/samples.jsonl`
- `artifacts/cleaning/discovery/patterns_top.jsonl`

### Step 2: Run Cleaning Pipeline

**Option A: Copy-paste the Colab version**

1. Copy the entire content from `notebooks/clean_logs_colab.py`
2. Paste into Colab cells (each `# CELL N:` section goes into a separate cell)
3. Run all cells sequentially (Runtime → Run all)

**Option B: Upload and run the script**

```python
# Upload clean_hf_logs.py to Colab
from google.colab import files
uploaded = files.upload()  # Upload scripts/clean_hf_logs.py

# Run it
!python clean_hf_logs.py --hf_repo chYassine/ait-fox-raw-v02 \
                         --out_dir datasets/ait/output/ait/cleaned \
                         --push_hf false
```

**Processing time:** ~20-30 minutes for 5.4M rows

### Step 3: Download Results

```python
# Download cleaned dataset
from google.colab import files
import shutil

# Zip the output
shutil.make_archive('cleaned_dataset', 'zip', 'datasets/ait/output/ait/cleaned')
files.download('cleaned_dataset.zip')

# Download artifacts
shutil.make_archive('cleaning_artifacts', 'zip', 'artifacts/cleaning')
files.download('cleaning_artifacts.zip')
```

## 📊 What Gets Created

### Cleaned Dataset Structure

```
datasets/ait/output/ait/cleaned/
├── dataset_info.json          # HF dataset metadata
├── data-00000-of-00001.arrow  # Compressed data
├── state.json                 # Dataset state
└── README.md                  # Documentation
```

### Artifacts

```
artifacts/cleaning/
├── discovery/                  # From Step 1
│   ├── hosts.csv
│   ├── log_types.csv
│   ├── host_logtype_crosstab.csv
│   ├── samples.jsonl
│   └── patterns_top.jsonl
├── stats.json                  # Dataset statistics
├── quality_report.md           # Human-readable report
├── samples/
│   └── sample_train.csv        # 100-row sample
└── redaction_salt.txt          # Keep this secret!
```

## 🔍 Verify Cleaning Quality

```python
from datasets import load_from_disk
import json

# Load cleaned dataset
ds = load_from_disk("datasets/ait/output/ait/cleaned")

print(f"Total rows: {len(ds):,}")
print(f"Columns: {ds.column_names}")

# Check sample
sample = ds.shuffle(seed=42).select(range(5))
for row in sample:
    print(f"\nHost: {row['host_sanitized']}")
    print(f"Type: {row['log_type_canonical']}")
    print(f"Time: {row['timestamp']}")
    print(f"Text: {row['text'][:100]}...")
    print(f"Canonical: {row['text_canonical'][:100]}...")

# Load statistics
with open("artifacts/cleaning/stats.json", "r") as f:
    stats = json.load(f)

print(f"\n📊 Statistics:")
print(f"  Timestamp parse rate: {stats['timestamp_parse_rate']:.1%}")
print(f"  Avg text length: {stats['avg_text_length']:.0f} chars")
print(f"  Top hosts: {list(stats['host_counts'].keys())[:5]}")
print(f"  Top log types: {list(stats['log_type_counts'].keys())[:5]}")
```

## 🎯 Common Customizations

### Change Repository

```python
HF_REPO = "your-username/your-dataset"
```

### Adjust Deduplication Window

In the config section:
```python
DEDUP_TIME_WINDOW_SECONDS = 5  # Default: 2
```

### Adjust Text Length Limits

```python
MAX_TEXT_LENGTH = 4096  # Default: 8192
MIN_TEXT_LENGTH = 10    # Default: 3
```

### Add Custom Log Type Mappings

```python
LOG_TYPE_CANONICAL = {
    # ... existing mappings ...
    "your_log_type": "canonical_name",
}

COARSE_LOG_TYPE = {
    # ... existing mappings ...
    "canonical_name": "category",
}
```

## 🐛 Troubleshooting

### Issue: Dataset too large for Colab RAM

**Solution:** Process in chunks

```python
# Load only a subset
dataset = load_dataset(HF_REPO, split="train[:10%]")  # First 10%
```

### Issue: Timestamp parse rate low (<95%)

**Solution:** Add custom timestamp patterns

```python
TIMESTAMP_PATTERNS.insert(0, (
    r'your_custom_pattern',
    '%Y-%m-%d %H:%M:%S'
))
```

### Issue: ImportError or package conflicts

**Solution:** Fresh restart

```python
# Restart runtime
import os
os.kill(os.getpid(), 9)

# Then reinstall
!pip install --no-cache-dir datasets==2.19.1 pandas tqdm
```

### Issue: Too many duplicates not removed

**Solution:** Increase dedup window or check hash function

```python
# Increase window
DEDUP_TIME_WINDOW_SECONDS = 10

# Or modify dedup key to include more fields
dedup_key = f"{host}:{log_type}:{epoch_bucket}:{text_hash}:{severity}"
```

## 📤 Push to HuggingFace

```python
# Set your HF token
from huggingface_hub import login
login()  # Follow prompts to paste token

# Load cleaned dataset
from datasets import load_from_disk
dataset = load_from_disk("datasets/ait/output/ait/cleaned")

# Push to HF
dataset.push_to_hub("your-username/ait-cleaned-logs", private=True)

print("✓ Pushed to HuggingFace!")
```

## 🔄 Incremental Cleaning (for updates)

If you need to clean new logs while keeping the same redaction salt:

```python
# First cleaning run
# Salt is saved to artifacts/cleaning/redaction_salt.txt

# Later runs: reuse the salt
import os
os.makedirs("artifacts/cleaning", exist_ok=True)

# Upload the saved redaction_salt.txt to Colab
from google.colab import files
uploaded = files.upload()  # Upload redaction_salt.txt

# Move it to the right location
!mv redaction_salt.txt artifacts/cleaning/

# Now run cleaning - it will reuse the same salt
# This ensures consistent redaction across dataset versions
```

## 📈 Performance Tips

### 1. Use batch processing

The pipeline already uses batching (batch_size=1000). For larger datasets:

```python
dataset = dataset.map(
    process_batch,
    batched=True,
    batch_size=2000,  # Increase if you have RAM
    num_proc=2,       # Use multiple cores (be careful in Colab)
)
```

### 2. Stream large datasets

```python
# For very large datasets (>10M rows)
dataset = load_dataset(HF_REPO, split="train", streaming=True)

# Process in chunks
chunk_size = 100000
for i, chunk in enumerate(dataset.take(chunk_size)):
    # Process chunk
    pass
```

### 3. Save intermediate checkpoints

```python
# After filtering
dataset.save_to_disk("checkpoint_filtered")

# After deduplication
dataset.save_to_disk("checkpoint_deduped")

# If Colab crashes, reload from checkpoint
dataset = load_from_disk("checkpoint_deduped")
```

## 🎓 Next Steps

After cleaning:

1. **Verify quality**: Check `artifacts/cleaning/quality_report.md`
2. **Sample inspection**: Review `artifacts/cleaning/samples/sample_train.csv`
3. **Phase 1 preprocessing**: Use cleaned data for template extraction and session building
4. **Model training**: Ready for log parsing, summarization, classification models

## 📚 Key Output Fields

| Field | Use Case |
|-------|----------|
| `text` | Training text (redacted, normalized) |
| `text_canonical` | Template extraction (Drain, Spell) |
| `timestamp` + `epoch_ms` | Time-series analysis |
| `rounded_ts_30m` | Session grouping |
| `log_type_canonical` | Classification target |
| `coarse_log_type` | High-level filtering |
| `structured_fields` | Feature extraction |
| `severity` | Priority filtering |
| `host_sanitized` | Multi-host analysis |

## ⚠️ Important Notes

1. **Redaction salt**: Keep `artifacts/cleaning/redaction_salt.txt` secret and consistent across runs
2. **Timestamp format**: All timestamps are UTC ISO8601
3. **Duplicates**: Deduplication uses 2-second windows; adjust if needed
4. **Text truncation**: Logs >8KB are truncated; check `text_was_truncated` flag
5. **Memory**: 5.4M rows needs ~6-8GB RAM in Colab (should fit in free tier)

## 📞 Support

- Script source: `scripts/clean_hf_logs.py`
- Colab version: `notebooks/clean_logs_colab.py`
- Dataset README: `datasets/ait/output/ait/cleaned/README.md`
- Phase 1 guide: `PHASE1_TRAINING_README.md`

## ✅ Checklist

- [ ] Step 1: Discovery completed, artifacts saved
- [ ] Step 2: Cleaning pipeline run, no errors
- [ ] Step 3: Quality verified (parse rate ≥95%, stats look good)
- [ ] Step 4: Downloaded cleaned dataset and artifacts
- [ ] Step 5: (Optional) Pushed to HuggingFace
- [ ] Ready for Phase 1 preprocessing!

---

**Time estimate**: 30-45 minutes total (including downloads)  
**Colab tier**: Free tier sufficient for 5.4M rows  
**Disk space**: ~2-3GB for outputs

