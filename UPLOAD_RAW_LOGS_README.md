# Upload Raw AIT Logs to Hugging Face as Dataset

This script reads raw AIT log files and creates a Hugging Face Dataset object with structured data columns.

## 🎯 Purpose

This script reads all raw AIT log files and creates a Hugging Face Dataset with structured columns:
- **content**: The actual log line text
- **path**: Source file path (e.g., `attacker_0/logs/apache_access/access.log`)
- **host**: Host name (e.g., `attacker_0`)
- **log_type**: Type of log (e.g., `apache_access`)
- **line_number**: Line number in the file

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

1. ✅ Reads all log lines from `datasets/ait/gather/`
2. ✅ Creates structured entries with content, path, host, log_type, and line_number
3. ✅ Builds a Hugging Face Dataset object
4. ✅ Uploads the Dataset to Hugging Face

### Dataset Structure

The uploaded dataset will have these columns:

| content | path | host | log_type | line_number |
|---------|------|------|----------|-------------|
| `2024-01-01 10:00:00 GET /index.html` | `attacker_0/logs/apache_access/access.log` | `attacker_0` | `apache_access` | `42` |
| `Connection established from 192.168.1.1` | `webserver/logs/suricata/log.pcap` | `webserver` | `suricata` | `100` |

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
  [1] Reading: access.log (2.5 MB)
      ✅ Read 15,234 lines
  [2] Reading: error.log (1.2 MB)
      ✅ Read 8,943 lines
  ...

======================================================================
CREATING DATASET
======================================================================
Total log entries: 2,456,789

Creating Dataset object...
✅ Dataset created with 2,456,789 entries
   Features: {'content': Value(dtype='string'), 'path': Value(dtype='string'), 
              'host': Value(dtype='string'), 'log_type': Value(dtype='string'),
              'line_number': Value(dtype='int64')}

======================================================================
UPLOADING TO HUGGING FACE
======================================================================
Uploading to: chYassine/ait-fox-raw-logs
This may take a while...

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

1. **Load the dataset**:
```python
from datasets import load_dataset

dataset = load_dataset("chYassine/ait-fox-raw-logs")
```

2. **Access the data**:
```python
# Access all log lines
for entry in dataset:
    print(f"Path: {entry['path']}")
    print(f"Content: {entry['content']}")
    print(f"Host: {entry['host']}")
    print(f"Log Type: {entry['log_type']}")
    print()

# Filter by host
attacker_logs = dataset.filter(lambda x: x['host'] == 'attacker_0')

# Filter by log type
apache_logs = dataset.filter(lambda x: x['log_type'] == 'apache_access')
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

