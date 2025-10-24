# Hugging Face Dataset Upload Guide

This guide explains how to upload the processed AIT Fox dataset to Hugging Face Hub while managing memory efficiently within a 12GB RAM limit.

## Prerequisites

1. **Hugging Face Account**: Create an account at [huggingface.co](https://huggingface.co)
2. **Access Token**: Generate a token from your Hugging Face settings
3. **Required Packages**: Install additional dependencies

## Installation

```bash
# Install additional requirements for uploading
pip install -r requirements_upload.txt
```

## Setup

1. **Set your Hugging Face token**:
   ```bash
   # Linux/Mac
   export HF_TOKEN=your_token_here
   
   # Windows
   set HF_TOKEN=your_token_here
   ```

2. **Verify your dataset files exist**:
   ```
   datasets/ait/output/ait/
   ├── ait_train.csv          # Required
   ├── ait_test.csv           # Required
   ├── ait_log_templates.json # Metadata
   └── ait_structured.csv     # Metadata
   ```

## Usage

### Option 1: Simple Upload (Recommended)

```bash
python upload_usage_example.py
```

### Option 2: Custom Upload

```python
from upload_to_huggingface import MemoryEfficientUploader

# Initialize uploader
uploader = MemoryEfficientUploader(
    dataset_path="datasets/ait/output/ait",
    repo_name="your-username/ait-fox-log-anomaly-dataset",
    chunk_size=3000,      # Adjust based on your RAM
    max_memory_gb=9.0     # Leave buffer for system
)

# Upload dataset
repo_url = uploader.upload_dataset(private=True)
```

## Memory Management

The uploader is designed to work within 12GB RAM constraints:

- **Chunk Processing**: Processes data in small chunks (default: 5000 rows)
- **Memory Monitoring**: Tracks memory usage and triggers cleanup when needed
- **Garbage Collection**: Forces cleanup to free memory between chunks
- **Configurable Limits**: Adjust chunk size and memory limits based on your system

### Recommended Settings for 12GB RAM:

```python
chunk_size=3000        # Conservative chunk size
max_memory_gb=9.0      # Leave 3GB buffer for system
```

## Configuration

Edit `upload_config.yaml` to customize:

- Repository name and settings
- Memory management parameters
- File processing options
- Feature schema definitions

## File Structure

The uploader creates a Hugging Face dataset with:

```
your-repo/
├── train/              # Training split
├── test/               # Test split
├── metadata/
│   ├── ait_log_templates.json
│   ├── ait_structured.csv
│   └── ait_templates.csv
└── README.md           # Auto-generated
```

## Dataset Schema

The uploaded dataset includes these features:

- `SessionId`: Unique session identifier
- `EventSequence`: Sequence of event IDs
- `SequenceLength`: Length of the sequence
- `Label`: Binary label (0=Normal, 1=Anomaly)
- `IsAttack`: Boolean attack indicator
- `AttackLabels`: Specific attack type labels
- `TextSequence`: Human-readable log sequence

## Monitoring Progress

The uploader provides detailed logging:

```
2025-01-XX XX:XX:XX - INFO - Processing train dataset: ait_train.csv
2025-01-XX XX:XX:XX - INFO - Processed 15000 rows, Memory: 8.45 GB
2025-01-XX XX:XX:XX - INFO - Uploading dataset to Hugging Face Hub
2025-01-XX XX:XX:XX - INFO - Dataset uploaded successfully
```

## Troubleshooting

### Memory Issues

If you encounter memory errors:

1. **Reduce chunk size**:
   ```python
   chunk_size=1000  # Smaller chunks
   ```

2. **Lower memory limit**:
   ```python
   max_memory_gb=7.0  # More conservative
   ```

3. **Close other applications** to free RAM

### Upload Errors

1. **Check HF token**: Ensure `HF_TOKEN` is set correctly
2. **Verify repository name**: Must be unique and follow HF naming conventions
3. **Check internet connection**: Upload requires stable connection
4. **Verify file permissions**: Ensure read access to dataset files

### Performance Tips

1. **Use SSD storage** for faster file I/O
2. **Close unnecessary applications** before uploading
3. **Use wired internet** for more stable upload
4. **Monitor system resources** during upload

## Example Output

```
🚀 Starting upload to: your-username/ait-fox-log-anomaly-dataset
✅ Prerequisites check passed!
📊 Processing train dataset: ait_train.csv
📊 Processing test dataset: ait_test.csv
📊 Uploading dataset to Hugging Face Hub
🎉 UPLOAD COMPLETED SUCCESSFULLY!
📊 Dataset URL: https://huggingface.co/datasets/your-username/ait-fox-log-anomaly-dataset
```

## Next Steps

After successful upload:

1. **Visit your dataset page** on Hugging Face
2. **Update the README** with dataset description
3. **Add tags** for better discoverability
4. **Share with collaborators** if needed
5. **Use in your projects** with the HF datasets library

## Support

If you encounter issues:

1. Check the logs for detailed error messages
2. Verify all prerequisites are met
3. Ensure sufficient system resources
4. Check Hugging Face Hub status
