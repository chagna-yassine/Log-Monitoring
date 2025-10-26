# Upload Raw AIT Logs to Hugging Face

This script uploads raw AIT log files directly to Hugging Face as a dataset, without any processing or sequence generation.

## 🎯 Purpose

Instead of processing logs in chunks (which creates many files and hits rate limits), this script uploads the original raw log files as-is to Hugging Face.

## 📋 Prerequisites

1. **Hugging Face Token**: Get your token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. **Extracted AIT Dataset**: The fox dataset should be extracted in `datasets/ait/`

## 🚀 Usage

### Run the Script

```bash
python scripts/upload_raw_logs.py
```

### What It Will Ask For

1. **Hugging Face Token**: Enter your personal access token
2. **Repository Name**: Enter the repository name (e.g., `username/ait-fox-raw-logs`)

### What It Does

1. ✅ Reads raw log files from `datasets/ait/gather/`
2. ✅ Uploads each log file preserving the directory structure
3. ✅ Creates a dataset repository on Hugging Face
4. ✅ Uploads metadata files (dataset.yml if available)

### Output Structure

The uploaded dataset will have this structure:

```
ait-fox-raw-logs/
├── raw_logs/
│   ├── {host}/
│   │   └── logs/
│   │       ├── {log_type}/
│   │       │   └── {log_file}
│   │       └── ...
│   └── ...
├── dataset.yml (if available)
└── upload_summary.txt
```

## 📊 Example

```bash
$ python scripts/upload_raw_logs.py

======================================================================
UPLOAD RAW AIT LOGS TO HUGGING FACE
======================================================================
======================================================================
HUGGING FACE UPLOAD SETUP
======================================================================

To upload raw logs to Hugging Face, you need:
1. A Hugging Face token (get from: https://huggingface.co/settings/tokens)
2. A repository name (e.g., 'username/ait-raw-logs')

Enter your Hugging Face token: hf_xxxxx...
✅ Token received!

Enter Hugging Face repository name: chYassine/ait-fox-raw-logs

Repository: chYassine/ait-fox-raw-logs
Token: ✅ Set
✅ Logged in to Hugging Face

======================================================================
PROCESSING AND UPLOADING RAW LOGS
======================================================================

Found 30 hosts

Processing host: attacker_0
  Found 152 log files
  [1] Uploading: raw_logs/attacker_0/logs/apache_access/access.log (2.5 MB)
      ✅ Uploaded successfully
  [2] Uploading: raw_logs/attacker_0/logs/apache_error/error.log (1.2 MB)
      ✅ Uploaded successfully
  ...

======================================================================
UPLOAD COMPLETE!
======================================================================
✅ Repository: https://huggingface.co/datasets/chYassine/ait-fox-raw-logs
✅ Total files uploaded: 4500
✅ Total hosts: 30

Dataset available at:
  https://huggingface.co/datasets/chYassine/ait-fox-raw-logs
```

## ✨ Advantages

- ✅ **No Rate Limits**: Uploads raw files without processing
- ✅ **Simple**: Just raw log files, no complex structures
- ✅ **Fast**: Direct upload without chunking or sequencing
- ✅ **Complete**: All log files from all hosts
- ✅ **Preserves Structure**: Maintains original directory layout

## 🔄 Using the Dataset Later

Once uploaded, you can:

1. **Load the raw logs**:
```python
from datasets import load_dataset

dataset = load_dataset("chYassine/ait-fox-raw-logs")
```

2. **Process locally** when you have time/resources:
```python
# Download and process locally
logs = dataset['raw_logs']
# ... your processing logic ...
```

3. **Share with others** without needing to download 26GB from Zenodo

## 📝 Notes

- The script preserves the original file structure
- Large files may take time to upload
- You can pause and resume - already uploaded files are skipped
- The upload summary shows what was uploaded

## 🆘 Troubleshooting

### Rate Limit Errors
If you get rate limit errors, wait an hour and run again. The script will skip already-uploaded files.

### Network Errors
If upload fails for a specific file, the script continues with other files. You can run it again to retry failed uploads.

### Token Errors
Make sure your token has "Write" permissions enabled in your Hugging Face settings.

