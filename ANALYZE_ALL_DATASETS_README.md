# Comprehensive AIT Dataset Analysis Script

This script provides comprehensive analysis of all AIT datasets uploaded to Hugging Face. It loads multiple datasets, analyzes them thoroughly, and displays detailed statistics and sample logs.

## Features

### 📊 Statistics Provided

1. **Total Log Entries**: Count across all loaded datasets
2. **Dataset Breakdown**: Number of logs per dataset
3. **Host Analysis**: 
   - Total unique hosts
   - Top hosts by log count
   - Hosts with most log type variety
4. **Log Type Analysis**:
   - Distribution of all log types
   - Percentage breakdown
5. **Content Analysis**:
   - Text vs binary file breakdown
   - Content length statistics (min, max, average, percentiles)
   - Binary file size analysis
6. **Cross-Analysis**:
   - Host vs log type combinations
   - Detailed breakdown per host

### 📋 Sample Logs Displayed

1. **By Log Type**: First example from each log type
2. **By Host**: First example from top 10 hosts
3. **Random Diversity**: 5 random samples ensuring diversity in type and host

## Requirements

Install the required packages:

```bash
pip install datasets huggingface_hub pandas
```

Or use the existing upload requirements:

```bash
pip install -r requirements_upload.txt
```

## Usage

### Basic Usage (All Default Datasets)

```bash
python scripts/analyze_all_ait_datasets.py
```

This will analyze all 8 AIT datasets:
- `chYassine/ait-wilson-raw-v01`
- `chYassine/ait-wheeler-raw-v01`
- `chYassine/ait-wardbeck-raw-v01`
- `chYassine/ait-shaw-raw-v01`
- `chYassine/ait-santos-raw-v01`
- `chYassine/air-russellmitchell-raw-v01`
- `chYassine/ait-harrison-raw-v01`
- `chYassine/ait-fox-raw-v02`

### List Default Datasets

```bash
python scripts/analyze_all_ait_datasets.py --list
```

### Custom Datasets

```bash
python scripts/analyze_all_ait_datasets.py --repos username/dataset1 username/dataset2
```

## Output

### Console Output

The script displays comprehensive analysis directly to the console:

```
================================================================================
  COMPREHENSIVE AIT DATASET ANALYSIS
================================================================================
Analysis started at: 2024-01-XX XX:XX:XX

================================================================================
  LOADING DATASETS
================================================================================
Loading: chYassine/ait-wilson-raw-v01
✅ Successfully loaded: 1,234,567 entries
  Columns: ['content', 'path', 'host', 'log_type', 'line_number', 'is_binary', 'file_size_mb']

...

================================================================================
  BASIC STATISTICS
================================================================================
Total Log Entries: 15,432,109
Total Datasets: 8

--------------------------------------------------------------------------------
  Host Analysis
--------------------------------------------------------------------------------
Total Unique Hosts: 120
Top 15 Hosts by Log Count:
  host1.example.com: 2,345,678 (15.2%)
  host2.example.com: 1,876,543 (12.2%)
  ...

--------------------------------------------------------------------------------
  Log Type Analysis
--------------------------------------------------------------------------------
Total Unique Log Types: 8
Log Types Distribution:
  apache_access: 8,765,432 (56.8%)
  apache_error: 3,456,789 (22.4%)
  syslog: 1,234,567 (8.0%)
  ...

--------------------------------------------------------------------------------
  Sample Log Entries
--------------------------------------------------------------------------------

📋 Log Type: apache_access
   Content: 192.168.1.1 - - [01/Jan/2024:12:00:00 +0000] "GET /index.html HTTP/1.1" 200 1234
   Host: host1.example.com
   Path: /var/log/apache2/access.log
```

### Saved Output

The script also saves a JSON summary:

**File**: `ait_datasets_analysis_summary.json`

```json
{
  "analysis_timestamp": "2024-01-XXTXX:XX:XX",
  "total_logs": 15432109,
  "num_datasets": 8,
  "dataset_breakdown": {
    "chYassine/ait-wilson-raw-v01": 1234567,
    "chYassine/ait-wheeler-raw-v01": 987654,
    ...
  },
  "columns": ["content", "path", "host", "log_type", "line_number", "is_binary", "file_size_mb"],
  "num_hosts": 120,
  "num_log_types": 8,
  "num_text_logs": 15300000,
  "num_binary_files": 132109,
  "host_distribution": {
    "host1.example.com": 2345678,
    "host2.example.com": 1876543,
    ...
  },
  "log_type_distribution": {
    "apache_access": 8765432,
    "apache_error": 3456789,
    ...
  }
}
```

## Dataset Structure

The script expects datasets with the following columns:

- `content`: The log content/text
- `path`: File path where the log came from
- `host`: Host/server name
- `log_type`: Type of log (e.g., apache_access, syslog, etc.)
- `line_number`: Line number in the original file
- `is_binary`: Boolean flag for binary files
- `file_size_mb`: Size in MB (for binary files)

## Examples

### Example 1: Quick Overview

```bash
python scripts/analyze_all_ait_datasets.py
```

Get a comprehensive overview of all 8 datasets.

### Example 2: Specific Datasets

```bash
python scripts/analyze_all_ait_datasets.py --repos chYassine/ait-fox-raw-v02 chYassine/ait-wilson-raw-v01
```

Analyze only Fox and Wilson datasets.

### Example 3: Custom Datasets

```bash
python scripts/analyze_all_ait_datasets.py --repos myusername/my-dataset
```

Analyze your own custom dataset.

## Key Insights Provided

### Host Patterns
- Which hosts generate the most logs
- Hosts with diverse log types
- Host-specific log type breakdowns

### Log Type Distribution
- Most common log types
- Percentage distribution
- Samples from each type

### Content Characteristics
- Text vs binary ratio
- Content length statistics
- Binary file sizes

### Cross-Dimensional Analysis
- Which hosts generate which log types
- Host-log type combinations
- Diversification of hosts

## Troubleshooting

### Dataset Not Found

```
❌ Failed to load: DatasetNotFound
```

**Solution**: 
1. Check the repository name is correct
2. Ensure you're logged in to Hugging Face (set `HUGGINGFACE_HUB_TOKEN`)
3. Verify the dataset is public or you have access

### Memory Issues

For very large datasets, you might encounter memory issues.

**Solution**: 
1. Analyze subsets of datasets
2. Use a machine with more RAM
3. Process datasets individually

### Authentication Issues

```
401 Client Error: Unauthorized
```

**Solution**: 
1. Set up your Hugging Face token:
   ```bash
   python scripts/setup_hf_token.py
   ```
2. Or manually set environment variable:
   ```bash
   set HUGGINGFACE_HUB_TOKEN=your_token_here
   ```

## Performance Notes

- **First Run**: May take time to download datasets if not cached
- **Cached Datasets**: Subsequent runs are much faster
- **Memory Usage**: Depends on total dataset size
- **Processing Time**: Typically 2-10 minutes for all 8 datasets

## Integration with Other Scripts

This analysis script complements:

1. **Upload Scripts**: After uploading, analyze the datasets
   ```bash
   python scripts/upload_raw_logs.py  # Upload
   python scripts/analyze_all_ait_datasets.py  # Analyze
   ```

2. **Individual Analysis**: Compare with dataset-specific analysis
   ```bash
   python scripts/analyze_hf_dataset.py --repo chYassine/ait-fox-raw-v02
   ```

3. **Preprocessing**: Use insights to plan preprocessing
   ```bash
   python scripts/preprocess_ait.py
   ```

## Next Steps

After analyzing:

1. **Review Insights**: Check host and log type distributions
2. **Plan Preprocessing**: Identify which hosts/types to focus on
3. **Check Samples**: Verify data quality from samples shown
4. **Document Findings**: Use the JSON summary for reports
5. **Proceed with Training**: Use insights to configure training

## See Also

- `DATASET_ANALYSIS_README.md` - Individual dataset analysis
- `UPLOAD_RAW_LOGS_README.md` - Upload process documentation
- `scripts/analyze_hf_dataset.py` - Single dataset analysis
- `HUGGINGFACE_TOKEN_SETUP.md` - Token setup guide

---

**Created**: January 2024  
**Version**: 1.0.0  
**Status**: Production Ready

