# Upload Status

## ✅ What's Done

1. **CSV Saved**: `datasets/ait/output/ait_raw_logs_dataset.csv`
   - **5,465,264 entries** from AIT fox dataset
   - All log files processed
   - Binary files recorded as metadata
   - Large files sampled/streamed

## 📊 Dataset Contents

- **Total entries**: 5,465,264
- **Columns**: 
  - `content`: Log line text (or `[BINARY FILE: filename]` for binary)
  - `path`: File path within dataset
  - `host`: Host name (e.g., `attacker_0`, `webserver`)
  - `log_type`: Type of log (e.g., `apache_access`, `suricata`)
  - `line_number`: Line number in file (0 for binary files)
  - `is_binary`: Whether file is binary (True/False)
  - `file_size_mb`: File size in MB (for binary files)

## 🎯 Next Steps

### Option 1: Upload Now (Recommended)

Run the upload script:
```bash
python scripts/upload_csv_to_hf.py
```

Enter the repository name (e.g., `chYassine/ait-fox-raw-logs`)

### Option 2: Upload Manually

If you prefer to upload manually:

```python
from datasets import Dataset

# Load the CSV
dataset = Dataset.from_csv("datasets/ait/output/ait_raw_logs_dataset.csv")

# Upload to Hugging Face
dataset.push_to_hub("chYassine/ait-fox-raw-logs")
```

### Option 3: Wait for Rate Limit Reset

If you hit a rate limit, wait 1 hour and try again.

## 📁 Files

- `datasets/ait/output/ait_raw_logs_dataset.csv` - The complete dataset (5.4M entries)
- `scripts/upload_csv_to_hf.py` - Upload script for the CSV

## ✨ Features

- ✅ All log entries preserved
- ✅ Binary files recorded as metadata
- ✅ Path information included
- ✅ Host and log type metadata
- ✅ Memory-efficient processing

