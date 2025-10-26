# Dataset Analysis and Visualization

This script analyzes AIT datasets uploaded to Hugging Face and generates comprehensive visualizations and statistics.

## Features

The analysis script provides:

### 📊 Charts and Visualizations

1. **Host Distribution**: Top 20 hosts by log entry count
2. **Log Type Distribution**: Pie chart of log types
3. **Binary vs Text Logs**: Comparison of text logs vs binary files
4. **Content Length Distribution**: Histogram of log content character count
5. **File Size Distribution**: Histogram of binary file sizes
6. **Anomaly Distribution**: Normal vs anomaly log comparison
7. **Anomaly by Host**: Top 15 hosts with most anomalies
8. **Anomaly by Log Type**: Distribution of anomalies across log types

### 📋 Reports

1. **Dataset Statistics**: JSON file with all key metrics
2. **Sample Entries**: CSV file with first 10 entries
3. **Analysis Report**: Comprehensive text report with statistics

## Requirements

```bash
pip install datasets pandas matplotlib seaborn pyyaml huggingface_hub
```

Or use the existing requirements.txt.

## Usage

### Basic Usage

```bash
python scripts/analyze_hf_dataset.py --repo username/ait-fox-raw-logs
```

### Custom Output Directory

```bash
python scripts/analyze_hf_dataset.py --repo username/ait-fox-raw-logs --output my_analysis
```

## Example Outputs

All outputs are saved to the specified output directory (default: `analysis_output/`):

- `host_distribution.png` - Top hosts chart
- `log_type_distribution.png` - Log types pie chart
- `binary_vs_text.png` - Binary vs text comparison
- `content_length_distribution.png` - Content length histogram
- `file_size_distribution.png` - Binary file sizes
- `anomaly_distribution.png` - Normal vs anomaly bar chart
- `anomaly_by_host.png` - Top hosts with anomalies
- `anomaly_by_log_type.png` - Anomaly distribution by type
- `dataset_statistics.json` - All statistics in JSON format
- `sample_entries.csv` - Sample dataset entries
- `analysis_report.txt` - Comprehensive text report

## Anomaly Detection

The script uses keyword-based anomaly detection. Anomalies are detected based on keywords like:
- attack, malicious, suspicious, intrusion, breach
- unauthorized, failed, denied, blocked
- nmap, sql injection, xss, ddos, botnet
- virus, malware, phishing, spam, hack
- And many more...

## Example Reports

### Dataset Statistics (JSON)

```json
{
  "total_entries": 5420000,
  "hosts": 15,
  "log_types": 8,
  "binary_files": 1234,
  "text_logs": 5418766,
  "normal_count": 5390000,
  "anomaly_count": 30000
}
```

### Analysis Report (Text)

```
Dataset Analysis Report: username/ait-fox-raw-logs
======================================================================

BASIC STATISTICS
----------------------------------------------------------------------
Total Entries: 5,420,000
Unique Hosts: 15
Unique Log Types: 8
Text Logs: 5,418,766
Binary Files: 1,234
Normal Logs: 5,390,000
Anomaly Logs: 30,000
Anomaly Rate: 0.55%

...
```

## Supported Datasets

This script works with any AIT dataset uploaded to Hugging Face with the following structure:

- `content`: Log content
- `path`: File path
- `host`: Host name
- `log_type`: Log type
- `line_number`: Line number
- `is_binary`: Binary flag
- `file_size_mb`: File size (for binary files)

## Tips

1. **Memory Management**: For very large datasets, consider using a smaller split or processing in chunks
2. **Custom Anomaly Keywords**: Edit the `attack_keywords` list in the script to customize detection
3. **Output Location**: Use `--output` to specify a custom output directory
4. **Public vs Private**: The repo must be accessible with your Hugging Face credentials

## Troubleshooting

### Dataset Not Found

```
❌ Failed to load dataset: DatasetNotFound
```

**Solution**: Make sure the repository name is correct and you're logged in to Hugging Face.

### Memory Issues

For very large datasets, you might encounter memory issues.

**Solution**: Process in smaller chunks or use a machine with more RAM.

### Import Errors

```
ModuleNotFoundError: No module named 'matplotlib'
```

**Solution**: Install required packages:
```bash
pip install matplotlib seaborn pandas
```

## Integration with Data Processing

This analysis script complements the upload scripts:

1. Upload dataset: `python scripts/upload_raw_logs.py`
2. Analyze dataset: `python scripts/analyze_hf_dataset.py --repo <repo>`
3. Review results: Check the output directory for charts and reports

## Next Steps

After analyzing the dataset:

1. Review the anomaly distribution to understand the dataset
2. Use the statistics to plan preprocessing steps
3. Identify hosts and log types with most anomalies
4. Create custom training/test splits based on insights
5. Use the visualizations for documentation and presentations

## See Also

- `UPLOAD_RAW_LOGS_README.md` - Upload scripts documentation
- `scripts/upload_raw_logs.py` - Upload Fox dataset
- `scripts/upload_harrison_logs.py` - Upload Harrison dataset
- `scripts/upload_russellmitchell_logs.py` - Upload Russell Mitchell dataset
- `scripts/upload_santos_logs.py` - Upload Santos dataset

