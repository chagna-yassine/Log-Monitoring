# Phase 1 Training Preprocessing Guide

This document describes the preprocessing pipeline for the **first training phase** of the log understanding base model, as specified in the training methodology.

## Overview

The first training phase builds a log understanding base model that captures:
- **Syntax**: Token-level parsing and template recognition
- **Templates**: Mapping between log text and event templates
- **Temporal Structure**: Short to medium-term relationships between log events
- **Multi-source Robustness**: Embeddings that work across hosts and log types

## Architecture Requirements

The preprocessing prepares data for a hybrid architecture:
- **Dense Transformer Layers**: Initial layers for stable token-level parsing
- **Soft Mixture-of-Experts (MoE)**: Efficient contextual aggregation across correlated lines
- **Session-level Embeddings**: Pooled outputs representing session context

## Training Objectives

The preprocessing prepares data for four simultaneous training objectives:

### 1. Masked Log Modeling
- **Purpose**: Teach token and parameter reconstruction
- **Method**: Randomly mask 15% of tokens, train model to predict them
- **Output**: Sequences with mask positions and original tokens

### 2. Next-Log Prediction
- **Purpose**: Enforce temporal relationships and short-term causality
- **Method**: Predict subsequent lines in the same session
- **Output**: Context windows with next-log labels

### 3. Contrastive Session Summarization
- **Purpose**: Compress multi-line sessions to fixed embeddings
- **Method**: Different views of the same session should map close in embedding space
- **Output**: Multiple views of sessions (time crops, sampled sequences)

### 4. Template-ID Classification
- **Purpose**: Strengthen mapping between templates and text representations
- **Method**: Auxiliary classification task for template identification
- **Output**: Log text with corresponding template IDs

## Data Sampling Strategy

### Context-Aware Batching
- **Grouping**: By host and recent time window
- **Rationale**: Events within a batch are more likely to be related
- **Implementation**: Sessions grouped by `host_logtype_timewindow`

### Curriculum Learning
- **Method**: Progressive sequence length increase
- **Structure**: Sequences organized into length buckets: [8, 16, 32, 64, 128, 256, 512]
- **Rationale**: Start with short sequences to stabilize learning, gradually increase

### Distribution
- **Pretraining**: Majority normal, no anomaly oversampling
- **Fine-tuning**: Anomaly oversampling or class weighting applied as needed

## Usage

### Option 1: From HuggingFace Dataset

If you have the AIT dataset uploaded to HuggingFace:

```bash
python scripts/preprocess_phase1_training.py <hf_repo_name>
```

Example:
```bash
python scripts/preprocess_phase1_training.py chYassine/ait-fox-raw-logs
```

### Option 2: From Local Processed Data

If you have already run the basic preprocessing:

```bash
# First, run basic preprocessing
python scripts/preprocess_ait.py

# Then, run Phase 1 preprocessing
python scripts/preprocess_phase1_training.py
```

## Configuration

Edit `config.yaml` to customize Phase 1 preprocessing:

```yaml
phase1_training:
  time_window_minutes: 30      # Session grouping window
  max_sequence_length: 512       # Maximum sequence length
  min_sequence_length: 4        # Minimum sequence length
  curriculum_buckets: [8, 16, 32, 64, 128, 256, 512]
  mask_probability: 0.15         # Masking probability
  next_log_window_size: 5       # Context window size
```

## Output Files

The preprocessing generates the following files in `datasets/ait/output/ait/phase1_training/`:

### CSV Files

1. **sessions.csv**
   - Session-level data with metadata
   - Columns: `session_id`, `template_ids`, `log_texts`, `sequence_length`, `curriculum_bucket`, `host`, `log_type`, `time_window`, `label`, `is_attack`

2. **masked_lm.csv**
   - Masked log modeling samples
   - Columns: `session_id`, `template_ids`, `masked_positions`, `num_masked`

3. **next_log.csv**
   - Next-log prediction samples
   - Columns: `session_id`, `context_ids`, `context_texts`, `next_template_id`, `next_text`

4. **contrastive.csv**
   - Contrastive session summarization samples
   - Columns: `session_id`, `view1_ids`, `view1_texts`, `view2_ids`, `view2_texts`

5. **template_classification.csv**
   - Template-ID classification samples
   - Columns: `session_id`, `log_text`, `template_id`, `position`

6. **metadata.json**
   - Processing metadata and statistics
   - Includes dataset info, sample counts, configuration

### HuggingFace Datasets (if created)

Optimized HuggingFace datasets are also created for efficient training:
- `hf_sessions/`
- `hf_masked_lm/`
- `hf_next_log/`
- `hf_contrastive/`
- `hf_template_classification/`

## Data Structure

### Session Grouping

Sessions are created using context-aware grouping:

```
Session ID Format: {host}_{log_type}_{time_window}

Example:
- webserver_apache_access_2022-01-18_11:30
- attacker_0_suricata_2022-01-18_12:00
- mailserver_auth_2022-01-18_11:45
```

### Sequence Metadata

Each session includes:
- **Host**: Source hostname (e.g., `webserver`, `attacker_0`)
- **Log Type**: Type of log (e.g., `apache_access`, `suricata`, `auth`)
- **Time Window**: Rounded timestamp window (e.g., `2022-01-18_11:30`)
- **Sequence Length**: Number of log entries in session
- **Curriculum Bucket**: Length bucket for curriculum learning
- **Labels**: Attack labels if available

## Training Data Statistics

After preprocessing, you'll see statistics like:

```
✅ Built 125,340 session sequences
  Average sequence length: 43.2
  Min/Max length: 4/512
  Curriculum buckets distribution:
    8: 15,234 sessions
    16: 28,456 sessions
    32: 35,678 sessions
    64: 24,567 sessions
    128: 15,234 sessions
    256: 4,567 sessions
    512: 1,464 sessions
```

## Integration with Training

The preprocessed data is ready for training with:

1. **PyTorch DataLoaders**: Load CSV files or HuggingFace datasets
2. **Context-aware batching**: Use `host` and `time_window` for batch construction
3. **Curriculum learning**: Sample from buckets progressively
4. **Multi-objective training**: Load different datasets for each objective

### Example Training Loop Structure

```python
from datasets import load_from_disk

# Load datasets
sessions_ds = load_from_disk('datasets/ait/output/ait/phase1_training/hf_sessions')
masked_lm_ds = load_from_disk('datasets/ait/output/ait/phase1_training/hf_masked_lm')
next_log_ds = load_from_disk('datasets/ait/output/ait/phase1_training/hf_next_log')

# Create data loaders with context-aware batching
# Group by host and time_window for related events
# Sample from curriculum buckets progressively
```

## Memory Considerations

The preprocessing handles large datasets efficiently:
- Processes data in chunks
- Creates HuggingFace datasets for streaming
- Saves intermediate results
- Provides CSV fallback if HF datasets fail

## Next Steps

After Phase 1 preprocessing:

1. **Train Base Model**: Use the preprocessed data to train the log understanding model
2. **Distillation**: Use base model embeddings for distillation into smaller models
3. **Supervised Fine-tuning**: Use session embeddings for anomaly detection

## References

- Training methodology: First training phase specification
- Dataset: [AIT-LDS v2.0](https://zenodo.org/records/5789064)
- Model architecture: Hybrid transformer with soft MoE

## Troubleshooting

### Missing Data Columns
If you get errors about missing columns, ensure:
- Basic preprocessing has been run: `python scripts/preprocess_ait.py`
- Or use HuggingFace dataset with required columns

### Memory Issues
- Process smaller subsets of data
- Increase chunk size in processing
- Use HuggingFace streaming datasets

### Empty Sessions
- Check `min_sequence_length` in config
- Verify time window grouping is working
- Check input data has timestamps

## Support

For issues or questions:
1. Check preprocessing logs for specific errors
2. Verify configuration in `config.yaml`
3. Ensure input data is properly formatted

