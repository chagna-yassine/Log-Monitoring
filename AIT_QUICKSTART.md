# AIT Dataset Attack Extractor - Quick Start Guide

## 🚀 One-Minute Setup

```bash
# 1. Install dependencies
pip install -r requirements_ait.txt

# 2. Download and extract attack logs (e.g., from fox dataset)
python download_ait_attacks.py --name=fox.zip

# 3. Analyze the results
python analyze_ait_attacks.py fox_attacks.parquet
```

That's it! 🎉

## 📋 Command Reference

### Download Attack Logs

```bash
# Basic usage - smallest dataset (7.1 GB)
python download_ait_attacks.py --name=russellmitchell.zip

# Keep temporary files for inspection
python download_ait_attacks.py --name=santos.zip --no-cleanup

# Custom output filename
python download_ait_attacks.py --name=fox.zip --output=my_attacks.csv
```

### Available Datasets (sorted by size)

| Dataset               | Download Size | Command                                             |
|-----------------------|---------------|-----------------------------------------------------|
| russellmitchell.zip   | 7.1 GB        | `--name=russellmitchell.zip`                        |
| santos.zip            | 10.0 GB       | `--name=santos.zip`                                 |
| fox.zip               | 15.8 GB       | `--name=fox.zip`                                    |
| harrison.zip          | 16.8 GB       | `--name=harrison.zip`                               |
| wardbeck.zip          | 17.2 GB       | `--name=wardbeck.zip`                               |
| shaw.zip              | 17.6 GB       | `--name=shaw.zip`                                   |
| wheeler.zip           | 19.6 GB       | `--name=wheeler.zip`                                |
| wilson.zip            | 26.5 GB       | `--name=wilson.zip`                                 |

### Analyze Extracted Attacks

```bash
# Full analysis report
python analyze_ait_attacks.py fox_attacks.parquet

# Works with CSV too (but slower)
python analyze_ait_attacks.py fox_attacks.csv
```

## 📊 Output Files

After running the extractor, you get:

1. **`<dataset>_attacks.csv`** - Human-readable CSV format
2. **`<dataset>_attacks.parquet`** - Fast-loading binary format (recommended)
3. **`escalation_attacks.csv`** - Filtered privilege escalation logs
4. **`scan_attacks.csv`** - Filtered scanning logs
5. **`rce_attacks.csv`** - Filtered remote code execution logs

## 🐍 Python Usage

```python
import pandas as pd

# Load extracted attacks
df = pd.read_parquet('fox_attacks.parquet')

# Show first few rows
print(df.head())

# Count attacks by type
print(df['attack_labels'].value_counts())

# Get all privilege escalation events
escalations = df[df['attack_labels'].str.contains('escalate')]

# Export specific host's attacks
host_attacks = df[df['host'] == 'intranet_server']
host_attacks.to_csv('intranet_attacks.csv', index=False)
```

## 📈 Example Analysis Output

```
📊 AIT Attack Log Analysis
================================================================================

1️⃣ Basic Statistics
   Total attack log lines: 12,345
   Unique log files: 45
   Unique hosts: 8
   Log types: 12

2️⃣ Attack Label Distribution
   - scan                        : 5,234 (42.37%)
   - escalate                    : 2,103 (17.03%)
   - attacker_change_user        : 1,876 (15.19%)
   - rce                         : 1,234 (10.00%)
   - webshell_upload             :   456 ( 3.69%)
   - data_exfiltration           :   234 ( 1.90%)

3️⃣ Attacks by Host
   - web_server                  : 4,567 (37.00%)
   - intranet_server             : 3,210 (26.00%)
   - firewall                    : 2,134 (17.29%)
```

## ⚡ Pro Tips

1. **Start Small**: Use `russellmitchell.zip` (7.1 GB) for testing
2. **Use Parquet**: 10x faster loading than CSV
3. **Filter Early**: Use `df[df['attack_labels'].str.contains('escalate')]`
4. **Memory**: Large datasets may need 4-8 GB RAM
5. **Keep Files**: Use `--no-cleanup` to inspect raw logs later

## 🔍 DataFrame Columns

| Column          | Example                                          |
|-----------------|--------------------------------------------------|
| log_file        | `intranet_server/logs/audit/audit.log`           |
| host            | `intranet_server`                                |
| log_type        | `audit`                                          |
| line_number     | `1860`                                           |
| timestamp       | `2022-01-24 03:15:42`                            |
| log_content     | `type=USER_AUTH msg=audit(1642999060.603:...`    |
| attack_labels   | `attacker_change_user,escalate`                  |
| attack_rules    | `{"attacker_change_user": ["attacker.esca..."]"` |

## 🛠️ Troubleshooting

**Problem**: Download fails  
**Solution**: Check internet connection, retry with the same command

**Problem**: Not enough disk space  
**Solution**: Use `russellmitchell.zip` (smallest) or free up space

**Problem**: Memory error loading Parquet  
**Solution**: Use chunked loading:
```python
import pandas as pd
for chunk in pd.read_parquet('fox_attacks.parquet', chunksize=10000):
    process(chunk)
```

## 📚 Learn More

- Full documentation: `AIT_ATTACK_EXTRACTOR_README.md`
- Dataset info: https://zenodo.org/records/5789064
- Paper: Landauer et al. (2022) - IEEE TDSC

## 💡 Common Use Cases

### 1. Find All Privilege Escalations
```bash
python download_ait_attacks.py --name=santos.zip
python -c "
import pandas as pd
df = pd.read_parquet('santos_attacks.parquet')
escalations = df[df['attack_labels'].str.contains('escalate')]
print(escalations[['host', 'timestamp', 'log_content']])
"
```

### 2. Count Attacks Per Host
```bash
python -c "
import pandas as pd
df = pd.read_parquet('fox_attacks.parquet')
print(df['host'].value_counts())
"
```

### 3. Export Specific Attack Type
```bash
python -c "
import pandas as pd
df = pd.read_parquet('fox_attacks.parquet')
scans = df[df['attack_labels'].str.contains('scan')]
scans.to_csv('only_scans.csv', index=False)
"
```

---

**Need help?** Check the full README or open an issue!

