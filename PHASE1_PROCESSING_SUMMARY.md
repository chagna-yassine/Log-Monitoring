# Phase 1 Training Processing Summary

## Overview

This document summarizes the processing pipeline required for the **first training phase** of the log understanding base model using the AIT dataset from HuggingFace.

## Processing Pipeline

### Step 1: Data Source Selection

The preprocessing script supports two data sources:

#### Option A: HuggingFace Dataset (Recommended)
```bash
python scripts/preprocess_phase1_training.py <hf_repo_name>
```

**Advantages:**
- Direct access to preprocessed data
- No local preprocessing required
- Can work with any HuggingFace repository

**Example:**
```bash
python scripts/preprocess_phase1_training.py chYassine/ait-fox-raw-logs
```

#### Option B: Local Processed Data
```bash
# First run basic preprocessing
python scripts/preprocess_ait.py

# Then run Phase 1 preprocessing
python scripts/preprocess_phase1_training.py
```

**Advantages:**
- Works with existing local data
- Can customize preprocessing steps
- Useful for debugging

### Step 2: Context-Aware Session Grouping

The preprocessing creates **context-aware sessions** by grouping logs:

**Grouping Criteria:**
- **Host**: Same hostname (e.g., `webserver`, `attacker_0`)
- **Log Type**: Same log type (e.g., `apache_access`, `suricata`)
- **Time Window**: Same 30-minute window (configurable)

**Session ID Format:**
```
{host}_{log_type}_{time_window}

Examples:
- webserver_apache_access_2022-01-18_11:30
- attacker_0_suricata_2022-01-18_12:00
```

**Rationale:**
- Batches are constructed to group records by host and time window
- Events within a batch are more likely to be related
- Enables efficient context-aware batching during training

### Step 3: Session Sequence Building

For each session group:
1. **Sort by timestamp** (if available)
2. **Extract template IDs** from log entries
3. **Extract log text** for each entry
4. **Filter by length**: Only include sequences within `min_sequence_length` and `max_sequence_length`
5. **Assign curriculum bucket**: Categorize by sequence length for progressive training

**Curriculum Buckets:**
- 8, 16, 32, 64, 128, 256, 512 tokens
- Sequences are assigned to the smallest bucket that fits their length
- Enables curriculum learning: start with short sequences, gradually increase

### Step 4: Multi-Objective Data Preparation

The preprocessing prepares data for **four simultaneous training objectives**:

#### 4.1 Masked Log Modeling
**Purpose:** Teach token and parameter reconstruction

**Method:**
- Randomly mask 15% of tokens in each sequence
- Store mask positions and original tokens
- Model predicts masked tokens from context

**Output:**
- `masked_lm.csv`: Sequences with mask positions
- Fields: `session_id`, `template_ids`, `masked_positions`, `num_masked`

#### 4.2 Next-Log Prediction
**Purpose:** Enforce temporal relationships and short-term causality

**Method:**
- Create sliding windows of context (window size: 5)
- Predict next log entry in sequence
- Multiple samples per session

**Output:**
- `next_log.csv`: Context windows with next-log labels
- Fields: `session_id`, `context_ids`, `context_texts`, `next_template_id`, `next_text`

#### 4.3 Contrastive Session Summarization
**Purpose:** Compress multi-line sessions to fixed embeddings

**Method:**
- Create multiple views of the same session:
  - **Time crop**: First half vs. second half
  - **Sequence sample**: Full sequence vs. sampled (every other element)
- Different views should map close in embedding space

**Output:**
- `contrastive.csv`: Multiple views of sessions
- Fields: `session_id`, `view1_ids`, `view1_texts`, `view2_ids`, `view2_texts`

#### 4.4 Template-ID Classification
**Purpose:** Strengthen mapping between templates and text representations

**Method:**
- Create samples: log text → template ID
- Auxiliary classification task
- One sample per log entry

**Output:**
- `template_classification.csv`: Log text with template IDs
- Fields: `session_id`, `log_text`, `template_id`, `position`

### Step 5: Data Export

The preprocessing exports data in two formats:

#### CSV Files (Human-readable)
- `sessions.csv`: Complete session data with metadata
- `masked_lm.csv`: Masked log modeling samples
- `next_log.csv`: Next-log prediction samples
- `contrastive.csv`: Contrastive learning samples
- `template_classification.csv`: Template classification samples
- `metadata.json`: Processing statistics and configuration

#### HuggingFace Datasets (Optimized)
- `hf_sessions/`: Sessions dataset
- `hf_masked_lm/`: Masked LM dataset
- `hf_next_log/`: Next-log dataset
- `hf_contrastive/`: Contrastive dataset
- `hf_template_classification/`: Template classification dataset

**Benefits:**
- Efficient streaming for large datasets
- Native PyTorch integration
- Built-in batching and shuffling

## Configuration

Edit `config.yaml` to customize preprocessing:

```yaml
phase1_training:
  time_window_minutes: 30      # Session grouping window
  max_sequence_length: 512       # Maximum sequence length
  min_sequence_length: 4        # Minimum sequence length
  curriculum_buckets: [8, 16, 32, 64, 128, 256, 512]
  mask_probability: 0.15         # Masking probability (15%)
  next_log_window_size: 5       # Context window for next-log prediction
```

## Output Structure

```
datasets/ait/output/ait/phase1_training/
├── sessions.csv                    # Session-level data
├── masked_lm.csv                  # Masked log modeling
├── next_log.csv                   # Next-log prediction
├── contrastive.csv                # Contrastive learning
├── template_classification.csv     # Template classification
├── metadata.json                   # Processing metadata
├── hf_sessions/                   # HuggingFace datasets
├── hf_masked_lm/
├── hf_next_log/
├── hf_contrastive/
└── hf_template_classification/
```

## Training Integration

### Loading Data

```python
from datasets import load_from_disk
import pandas as pd

# Option 1: HuggingFace datasets (recommended)
sessions_ds = load_from_disk('datasets/ait/output/ait/phase1_training/hf_sessions')
masked_lm_ds = load_from_disk('datasets/ait/output/ait/phase1_training/hf_masked_lm')

# Option 2: CSV files
sessions_df = pd.read_csv('datasets/ait/output/ait/phase1_training/sessions.csv')
```

### Context-Aware Batching

Group batches by:
- **Host**: Same host for similar patterns
- **Time Window**: Recent time window for temporal context
- **Curriculum Bucket**: Same length bucket for efficient processing

### Curriculum Learning

Start training with:
1. **Short sequences** (bucket 8): Stabilize learning
2. **Progressively increase** to longer sequences
3. **Long sequences** (bucket 512): Full context understanding

### Multi-Objective Training

Train simultaneously on:
- **Masked LM loss**: Token reconstruction
- **Next-log loss**: Temporal prediction
- **Contrastive loss**: Session summarization
- **Template classification loss**: Template mapping

## Expected Output Statistics

For the AIT Fox dataset (~5.4M log entries):

```
✅ Built ~125,000 session sequences
  Average sequence length: ~43
  Min/Max length: 4/512
  
Curriculum buckets distribution:
  8: ~15,000 sessions
  16: ~28,000 sessions
  32: ~36,000 sessions
  64: ~25,000 sessions
  128: ~15,000 sessions
  256: ~5,000 sessions
  512: ~1,500 sessions

Training samples:
  Masked LM: ~125,000 samples
  Next-log: ~5,000,000 samples (many per session)
  Contrastive: ~12,500 samples (10% sampled)
  Template classification: ~5,400,000 samples (one per log entry)
```

## Key Features

### ✅ Context-Aware Grouping
- Sessions grouped by host, log type, and time window
- Enables efficient batching of related events

### ✅ Curriculum Learning Support
- Sequences organized into length buckets
- Progressive training from short to long sequences

### ✅ Multi-Objective Ready
- Data prepared for all four training objectives
- Can train on all objectives simultaneously

### ✅ Memory Efficient
- Processes data in chunks
- Creates streaming datasets
- Handles large datasets efficiently

### ✅ Flexible Data Sources
- Works with HuggingFace datasets
- Works with local processed data
- Automatic column detection

## Next Steps

After preprocessing:

1. **Train Base Model**: Use the preprocessed data with your transformer + MoE architecture
2. **Validate Objectives**: Check that all four objectives converge
3. **Extract Embeddings**: Generate session-level embeddings
4. **Distillation**: Use embeddings for distillation into smaller models
5. **Supervised Fine-tuning**: Use embeddings for anomaly detection

## Troubleshooting

### Missing Columns
- Ensure input data has: `host`, `timestamp`, `log_type`, `EventId`, `content`
- Or run basic preprocessing first: `python scripts/preprocess_ait.py`

### Memory Issues
- Reduce `max_sequence_length` in config
- Process smaller subsets
- Use HuggingFace streaming datasets

### Empty Sessions
- Check `min_sequence_length` in config
- Verify time window grouping
- Check input data quality

## References

- **Training Methodology**: First training phase specification
- **Dataset**: [AIT-LDS v2.0](https://zenodo.org/records/5789064)
- **Architecture**: Hybrid transformer with soft MoE
- **Implementation**: `scripts/preprocess_phase1_training.py`

