# Data Cleaning and Filtering Description

## Overview

This document describes the comprehensive data cleaning and filtering pipeline applied to the AIT-LDS log dataset (`chYassine/ait-fox-raw-v02`) to prepare it for log understanding tasks (parsing, summarization, classification). The pipeline transforms raw, heterogeneous log data into a clean, normalized, and enriched dataset suitable for machine learning model training.

## Cleaning Pipeline Stages

### 1. Schema Normalization
- **Purpose**: Establish consistent column names and data types across all log entries
- **Actions**: 
  - Standardized column names (text, host, log_type, timestamp, etc.)
  - Ensured consistent data types for all fields
  - Handled missing fields with appropriate defaults

### 2. Text Normalization
- **Purpose**: Standardize text encoding and formatting
- **Actions**:
  - Unicode normalization (NFC form)
  - Whitespace normalization (collapsed multiple spaces, trimmed)
  - Control character removal/replacement
  - Special character handling

### 3. Timestamp Parsing and Normalization
- **Purpose**: Extract and standardize timestamps across multiple formats
- **Actions**:
  - Parsed timestamps in various formats (ISO8601, syslog, Apache, etc.)
  - Converted all timestamps to UTC ISO8601 format
  - Created epoch millisecond representations
  - Handled missing or invalid timestamps gracefully

### 4. Host and Log Type Sanitization
- **Purpose**: Normalize host names and log type identifiers
- **Actions**:
  - Removed special characters from host names
  - Created canonical mappings for log types
  - Validated host and log type values against known patterns

### 5. Structured Field Extraction
- **Purpose**: Extract structured information from log entries
- **Actions**:
  - Parsed JSON fields where present
  - Extracted key-value pairs
  - Identified entities (IPs, URLs, file paths, etc.)
  - Log-type-specific parsing (Apache, Suricata, systemd, etc.)

### 6. Sensitive Data Redaction
- **Purpose**: Protect sensitive information while preserving log structure
- **Actions**:
  - Replaced IP addresses with placeholders (e.g., `<IP>`)
  - Redacted email addresses
  - Masked UUIDs and MAC addresses
  - Replaced sensitive tokens with standardized placeholders

### 7. Garbage Filtering
- **Purpose**: Remove corrupted, non-log, or unparseable entries
- **Heuristics Applied**:
  - **Length checks**: Removed entries shorter than 3 characters
  - **Repeated character detection**: Filtered entries where a single character appears >70% of the time
  - **Pattern matching**: Removed entries matching garbage patterns (short random strings, only non-printable ASCII, only control characters)
  - **Entropy analysis**: Identified and removed high-entropy random strings that don't represent valid log messages
  - **Regex-based filtering**: Removed entries matching specific corruption patterns

### 8. Exact Deduplication
- **Purpose**: Remove exact duplicate log entries within a time window
- **Actions**:
  - Identified exact duplicates based on text content
  - Applied time-window based deduplication to avoid removing legitimate repeated logs
  - Preserved the first occurrence of each duplicate

### 9. Canonicalization
- **Purpose**: Create normalized representations for template mining
- **Actions**:
  - Replaced variable values (IPs, numbers, paths) with placeholders
  - Generated canonical log templates
  - Created consistent patterns for similar log types

### 10. Temporal Enrichment
- **Purpose**: Add time-based features for analysis
- **Actions**:
  - Extracted hour of day
  - Extracted day of week
  - Created rounded timestamp buckets
  - Added temporal metadata fields

### 11. Severity Extraction
- **Purpose**: Identify and normalize log severity levels
- **Actions**:
  - Extracted severity from log content (INFO, WARNING, ERROR, etc.)
  - Normalized severity levels across different log formats
  - Created severity mappings for different log types

### 12. Label Normalization
- **Purpose**: Ensure consistent label formats
- **Actions**:
  - Normalized binary labels (0/1, normal/anomaly)
  - Validated label consistency
  - Handled missing labels appropriately

### 13. Quality Validation
- **Purpose**: Ensure data quality and consistency
- **Actions**:
  - Validated required fields are present
  - Checked data type consistency
  - Performed statistical quality checks
  - Generated quality reports

### 14. Final Garbage Filtering Pass
- **Purpose**: Additional filtering pass to catch remaining corrupted entries
- **Actions**:
  - Applied refined garbage detection heuristics
  - Filtered entries with suspicious character patterns (e.g., `E\`D`, `ITD`, `V9]'y`)
  - Removed entries that are clearly not log messages despite passing initial filters
  - Ensured no valid log entries are incorrectly removed using conservative regex patterns

## Dataset Statistics

### Input Dataset
- **Source**: `chYassine/ait-fox-raw-v02` (HuggingFace)
- **Initial Size**: ~5.4M log entries
- **Hosts**: 20 unique hosts
- **Log Types**: 11 primary log types (Suricata, Logstash, Apache, Journal, etc.)

### Output Dataset
- **Repository**: `chYassine/ait-cleaned-logs-v1` (HuggingFace)
- **Format**: HuggingFace Dataset (Arrow format)
- **Schema**: Normalized columns with consistent types
- **Quality**: Cleaned, deduplicated, and validated logs ready for training

## Key Features

1. **Multi-Format Support**: Handles various log formats (Apache, Suricata, systemd, JSON, etc.)
2. **Preservation of Structure**: Maintains log semantics while normalizing format
3. **Privacy-Aware**: Redacts sensitive information while preserving patterns
4. **Quality-Focused**: Multiple validation and filtering stages ensure high data quality
5. **Scalable**: Processes large datasets efficiently using batch processing

## Post-Cleaning Validation

After cleaning, the dataset undergoes:
- Token counting (using Qwen/Qwen2.5-72B-Instruct tokenizer) for cost estimation
- Final quality checks
- Statistical analysis
- Sample validation to ensure cleaning effectiveness

The cleaned dataset is now ready for use in log understanding model training, including tasks such as masked log modeling, next-log prediction, contrastive session summarization, and template-ID classification.
