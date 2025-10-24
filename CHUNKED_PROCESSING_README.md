# AIT-LDS Chunked Processing Pipeline

This pipeline processes large AIT datasets in manageable 10GB chunks, uploads each chunk to Hugging Face, and deletes local files to save storage space.

## Features

- **Memory Efficient**: Processes data in 10GB chunks to avoid memory overflow
- **Storage Saving**: Uploads processed chunks to Hugging Face and deletes local files
- **Continuous Processing**: Automatically continues with next chunk after upload
- **Attack Detection**: Enhanced attack pattern detection for better labeling
- **Synthetic Attacks**: Generates synthetic attack sequences if needed
- **Template Extraction**: Uses Drain algorithm for log template extraction
- **Sequence Building**: Creates meaningful log sequences for model training

## Usage

### 1. Prerequisites

```bash
# Install required packages
pip install huggingface_hub pandas pyyaml tqdm

# Download AIT dataset
python scripts/download_data_ait.py
```

### 2. Run Chunked Processing

```bash
# Run the chunked processing pipeline
python scripts/run_chunked_processing.py
```

### 3. Manual Processing

```bash
# Run directly
python scripts/preprocess_ait_chunked.py
```

## Configuration

Edit `chunked_config.yaml` to customize:

- **Chunk size**: Default 10GB per chunk
- **Sequence length**: Number of logs per sequence
- **Attack detection**: Keywords and patterns for attack detection
- **Hugging Face settings**: Repository name and settings

## Output Structure

Each chunk uploaded to Hugging Face contains:

```
chunk_1/
├── ait_chunk_1_structured.csv    # Parsed log entries
├── ait_chunk_1_sequences.csv    # Event sequences
├── ait_chunk_1_train.csv        # Training data
├── ait_chunk_1_test.csv         # Test data
├── ait_chunk_1_templates.csv    # Template counts
└── ait_chunk_1_mapping.json     # Template mapping

chunk_2/
├── ait_chunk_2_structured.csv
├── ait_chunk_2_sequences.csv
├── ait_chunk_2_train.csv
├── ait_chunk_2_test.csv
├── ait_chunk_2_templates.csv
└── ait_chunk_2_mapping.json
```

## Memory Management

The pipeline automatically:

- **Processes in chunks**: Never loads more than 10GB at once
- **Cleans up memory**: Deletes processed data after upload
- **Forces garbage collection**: Frees memory between chunks
- **Monitors memory usage**: Warns if approaching limits

## Attack Detection

Enhanced attack detection using:

1. **Keyword matching**: Searches for attack-related terms
2. **Host analysis**: Identifies suspicious host names
3. **Pattern recognition**: Detects failed logins, port scans, etc.
4. **Synthetic generation**: Creates attack sequences if needed

## Hugging Face Integration

- **Automatic upload**: Each chunk uploaded immediately after processing
- **Repository creation**: Creates dataset repository if it doesn't exist
- **Progress tracking**: Shows upload progress for each file
- **Error handling**: Continues processing even if upload fails

## Benefits

- **Storage Efficient**: Saves local storage by uploading immediately
- **Memory Safe**: Never exceeds available RAM
- **Resumable**: Can restart from any point
- **Scalable**: Handles datasets of any size
- **Cloud Ready**: Data available on Hugging Face for sharing

## Example Output

```
======================================================================
AIT-LDS CHUNKED PREPROCESSING PIPELINE
======================================================================

AIT-LDS Configuration:
  Dataset: fox
  Dataset Path: datasets/ait
  Output Path: datasets/ait/output/ait

Processing in chunks of 200,000 rows (~10GB each)

======================================================================
PROCESSING CHUNK 1
======================================================================
Creating sequences from chunk 1...
Chunk 1 sequence statistics:
  Total sequences: 13,333
  Normal sequences: 7,600 (57.00%)
  Attack sequences: 5,733 (43.00%)

Running Drain template extraction on chunk 1...
Converting sequences to text format for chunk 1...
Splitting chunk 1 into train/test sets...
  Saved ait_chunk_1_structured.csv: 200,000 rows
  Saved ait_chunk_1_sequences.csv: 13,333 rows
  Saved ait_chunk_1_train.csv: 5,320 rows
  Saved ait_chunk_1_test.csv: 8,013 rows
  Saved ait_chunk_1_templates.csv: 1,245 rows
  Saved ait_chunk_1_mapping.json: 1,245 entries

  Uploading chunk 1 to Hugging Face...
    Uploaded ait_chunk_1_structured.csv
    Uploaded ait_chunk_1_sequences.csv
    Uploaded ait_chunk_1_train.csv
    Uploaded ait_chunk_1_test.csv
    Uploaded ait_chunk_1_templates.csv
    Uploaded ait_chunk_1_mapping.json
  ✓ Chunk 1 uploaded successfully
  ✓ Chunk 1 processed and uploaded

======================================================================
CHUNKED PREPROCESSING COMPLETE!
======================================================================
Total chunks processed: 25
Total entries processed: 5,000,000
Data uploaded to: https://huggingface.co/datasets/username/ait-processed-data
```

## Troubleshooting

### Memory Issues
- Reduce `chunk_size_gb` in `chunked_config.yaml`
- Enable `force_garbage_collection`
- Monitor system memory usage

### Upload Failures
- Check internet connection
- Verify Hugging Face credentials
- Check repository permissions

### Processing Errors
- Verify AIT dataset is complete
- Check file permissions
- Review error logs for specific issues

## Next Steps

After processing:

1. **Download chunks**: Use Hugging Face datasets library
2. **Combine data**: Merge chunks if needed for training
3. **Train models**: Use processed sequences for model training
4. **Benchmark**: Run evaluation on test data
5. **Share results**: Upload results to Hugging Face
