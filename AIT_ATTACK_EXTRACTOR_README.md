# AIT Dataset Attack Log Extractor

A Python script to download AIT Log Dataset files from Zenodo and extract only attack-labeled logs into a pandas DataFrame for easy analysis.

## Features

- ✅ Downloads datasets directly from Zenodo
- ✅ Automatically extracts zip files
- ✅ Parses ground-truth labels to identify attack logs
- ✅ Creates structured DataFrame with attack metadata
- ✅ Saves results in both CSV and Parquet formats
- ✅ Progress bars for downloads and extraction
- ✅ Automatic cleanup of temporary files

## Installation

1. Install required dependencies:

```bash
pip install -r requirements_ait.txt
```

## Usage

### Basic Usage

Download and extract attack logs from a specific dataset:

```bash
python download_ait_attacks.py --name=fox.zip
```

This will:
1. Download `fox.zip` from Zenodo
2. Extract the archive
3. Parse all labels and extract attack logs
4. Save to `fox_attacks.csv` and `fox_attacks.parquet`
5. Clean up temporary files

### Custom Output File

Specify a custom output file:

```bash
python download_ait_attacks.py --name=harrison.zip --output=my_attacks.csv
```

### Keep Downloaded Files

If you want to keep the downloaded and extracted files for further analysis:

```bash
python download_ait_attacks.py --name=santos.zip --no-cleanup
```

## Available Datasets

| Dataset Name          | Size   | Simulation Period        | Attack Period           | Unpacked Size |
|-----------------------|--------|--------------------------|-------------------------|---------------|
| `fox.zip`             | 15.8 GB| 2022-01-15 to 2022-01-20 | 2022-01-18 11:59-13:15  | 26 GB         |
| `harrison.zip`        | 16.8 GB| 2022-02-04 to 2022-02-09 | 2022-02-08 07:07-08:38  | 27 GB         |
| `russellmitchell.zip` | 7.1 GB | 2022-01-21 to 2022-01-25 | 2022-01-24 03:01-04:39  | 14 GB         |
| `santos.zip`          | 10.0 GB| 2022-01-14 to 2022-01-18 | 2022-01-17 11:15-11:59  | 17 GB         |
| `shaw.zip`            | 17.6 GB| 2022-01-25 to 2022-01-31 | 2022-01-29 14:37-15:21  | 27 GB         |
| `wardbeck.zip`        | 17.2 GB| 2022-01-19 to 2022-01-24 | 2022-01-23 12:10-12:56  | 26 GB         |
| `wheeler.zip`         | 19.6 GB| 2022-01-26 to 2022-01-31 | 2022-01-30 07:35-17:53  | 30 GB         |
| `wilson.zip`          | 26.5 GB| 2022-02-03 to 2022-02-09 | 2022-02-07 10:57-11:49  | 39 GB         |

## Output Format

The script generates a pandas DataFrame with the following columns:

| Column           | Description                                          |
|------------------|------------------------------------------------------|
| `log_file`       | Relative path to the log file                        |
| `host`           | Hostname/server where the log was generated          |
| `log_type`       | Type of log (e.g., audit, apache, dns)               |
| `line_number`    | Line number in the original log file                 |
| `timestamp`      | Extracted timestamp (if available)                   |
| `log_content`    | Full content of the log line                         |
| `attack_labels`  | Comma-separated attack labels (e.g., scan, escalate) |
| `attack_rules`   | JSON string of labeling rules that triggered         |

### Example Output

```csv
log_file,host,log_type,line_number,timestamp,log_content,attack_labels,attack_rules
intranet_server/logs/audit/audit.log,intranet_server,audit,1860,audit(1642999060.603:2226):,type=USER_AUTH msg=audit(1642999060.603:2226): pid=27950 uid=33...,attacker_change_user,escalate,"{""attacker_change_user"": [""attacker.escalate.audit.su.login""], ""escalate"": [""attacker.escalate.audit.su.login""]}"
```

## Attack Types in Dataset

The datasets contain logs for the following attack stages:

| Attack Label         | Description                                  |
|----------------------|----------------------------------------------|
| `scan`               | Port scanning (nmap, WPScan, dirb)           |
| `webshell_upload`    | Malicious file upload (CVE-2020-24186)       |
| `password_cracking`  | Brute force attacks (John the Ripper)        |
| `escalate`           | Privilege escalation                         |
| `rce`                | Remote command execution                     |
| `data_exfiltration`  | Data theft (e.g., DNSteal)                   |
| `attacker_change_user` | User account switching                     |

## How It Works

1. **Download**: Fetches the dataset zip file from Zenodo
2. **Extract**: Unzips the archive to access `gather/` (logs) and `labels/` (ground truth)
3. **Label Parsing**: Reads JSON label files that map line numbers to attack types
4. **Log Extraction**: Reads log files and extracts only lines marked as attacks
5. **DataFrame Creation**: Structures the attack logs with metadata
6. **Save**: Exports to CSV (human-readable) and Parquet (efficient storage)
7. **Cleanup**: Removes temporary download and extraction directories

## Example Usage in Python

After running the script, you can load and analyze the results:

```python
import pandas as pd

# Load the extracted attack logs
df = pd.read_parquet('fox_attacks.parquet')

# Show basic statistics
print(f"Total attack log lines: {len(df)}")
print(f"Unique hosts affected: {df['host'].nunique()}")

# Analyze attack types
attack_distribution = df['attack_labels'].value_counts()
print("\nAttack Type Distribution:")
print(attack_distribution)

# Filter specific attack type
escalation_logs = df[df['attack_labels'].str.contains('escalate')]
print(f"\nPrivilege escalation events: {len(escalation_logs)}")

# Group by host
attacks_by_host = df.groupby('host').size().sort_values(ascending=False)
print("\nAttacks by Host:")
print(attacks_by_host)
```

## Directory Structure

After running with `--no-cleanup`, you'll see:

```
.
├── download_ait_attacks.py          # Main script
├── requirements_ait.txt             # Dependencies
├── ait_downloads/                   # Downloaded zip files
│   └── fox.zip
├── ait_extracted/                   # Extracted datasets
│   └── fox/
│       ├── gather/                  # All logs
│       ├── labels/                  # Ground truth labels
│       ├── processing/              # Label generation code
│       ├── rules/                   # Labeling rules
│       └── environment/             # Testbed setup
├── fox_attacks.csv                  # Output CSV
└── fox_attacks.parquet              # Output Parquet
```

## Performance Tips

- **Storage**: Make sure you have enough disk space (datasets are 7-27 GB compressed, 14-39 GB uncompressed)
- **Memory**: Loading large DataFrames may require 4-8 GB RAM
- **Speed**: Use Parquet format for faster loading in subsequent analyses
- **Parallel Processing**: To process multiple datasets, run separate instances of the script

## Troubleshooting

### Download Fails

If the download fails or is interrupted:
```bash
# The script will resume using the partial download if it exists
# Or delete and retry:
rm -rf ait_downloads/fox.zip
python download_ait_attacks.py --name=fox.zip
```

### Extraction Issues

If extraction fails:
```bash
# Clean up and retry
rm -rf ait_extracted/fox
python download_ait_attacks.py --name=fox.zip
```

### No Attack Logs Found

This usually means the dataset structure is different than expected. Check:
```bash
ls -la ait_extracted/fox/
# Should contain: gather/, labels/, dataset.yml
```

## Citation

If you use this script or the AIT dataset in your research, please cite:

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

## License

This script is provided as-is for use with the AIT Log Dataset V2.0, which is licensed under CC-BY-NC-SA-4.0.

## Support

For issues with:
- **This script**: Open an issue in your repository
- **The dataset**: Contact the dataset authors or visit https://zenodo.org/records/5789064

