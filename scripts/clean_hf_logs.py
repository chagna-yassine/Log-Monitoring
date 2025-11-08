#!/usr/bin/env python3
"""
HuggingFace Log Cleaning Pipeline
Cleans raw logs for parsing, summarization, and classification tasks.
"""

import os
import json
import re
import hashlib
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm
import argparse

# ============================================================================
# CONFIGURATION
# ============================================================================

class CleaningConfig:
    """Configuration for log cleaning pipeline"""
    
    # Paths
    OUT_DIR = "datasets/ait/output/ait/cleaned"
    ARTIFACTS_DIR = "artifacts/cleaning"
    
    # Processing
    MAX_TEXT_LENGTH = 8192
    MIN_TEXT_LENGTH = 3
    SAMPLE_ROWS_PER_SPLIT = 100
    
    # Deduplication
    DEDUP_TIME_WINDOW_SECONDS = 2
    NEAR_DEDUP_THRESHOLD = 0.85
    
    # Redaction
    REDACTION_SALT_FILE = "artifacts/cleaning/redaction_salt.txt"
    
    # Timestamp rounding windows (minutes)
    TIME_WINDOWS = [1, 5, 30]
    
    # Canonical log type mappings
    LOG_TYPE_CANONICAL = {
        # web
        "apache": "apache2",
        "apache_access": "apache2",
        "httpd": "apache2",
        "nginx": "nginx",
        "web": "apache2",
        
        # network/IDS
        "suricata": "suricata",
        "ids": "suricata",
        "nids": "suricata",
        
        # auth
        "auth": "auth",
        "sshd": "auth",
        "pam": "auth",
        
        # system
        "syslog": "journal",
        "system": "journal",
        "systemd": "journal",
        
        # mail
        "exim": "exim4",
        "mail": "exim4",
        "postfix": "postfix",
        
        # dns
        "dns": "dns",
        "bind": "dns",
        
        # audit
        "auditd": "audit",
        
        # monitoring
        "logstash": "logstash",
        "elasticsearch": "logstash",
        
        # application
        "horde": "horde",
    }
    
    # Coarse log type categorization
    COARSE_LOG_TYPE = {
        "apache2": "web",
        "nginx": "web",
        "suricata": "network",
        "auth": "auth",
        "journal": "system",
        "exim4": "mail",
        "postfix": "mail",
        "horde": "mail",
        "dns": "network",
        "audit": "system",
        "logstash": "monitoring",
        "dnsteal": "network",
        "ait.aecid.attacker.wpdiscuz": "attack",
        "downloads": "application",
        "redis": "database",
    }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ensure_dirs():
    """Create output directories"""
    os.makedirs(CleaningConfig.OUT_DIR, exist_ok=True)
    os.makedirs(CleaningConfig.ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(f"{CleaningConfig.ARTIFACTS_DIR}/samples", exist_ok=True)


def get_redaction_salt() -> str:
    """Get or create stable salt for redaction"""
    salt_file = CleaningConfig.REDACTION_SALT_FILE
    os.makedirs(os.path.dirname(salt_file), exist_ok=True)
    
    if os.path.exists(salt_file):
        with open(salt_file, 'r') as f:
            return f.read().strip()
    
    salt = hashlib.sha256(str(datetime.now().timestamp()).encode()).hexdigest()[:16]
    with open(salt_file, 'w') as f:
        f.write(salt)
    return salt


REDACTION_SALT = None  # Will be initialized


def stable_hash(value: str, prefix: str = "") -> str:
    """Create stable hash for redaction"""
    global REDACTION_SALT
    if REDACTION_SALT is None:
        REDACTION_SALT = get_redaction_salt()
    
    salted = f"{REDACTION_SALT}:{value}"
    hash_val = hashlib.sha1(salted.encode()).hexdigest()[:8]
    return f"{prefix}{hash_val}" if prefix else hash_val


# ============================================================================
# TEXT NORMALIZATION
# ============================================================================

def normalize_unicode(text: str) -> str:
    """Normalize unicode to NFC form"""
    if not text:
        return ""
    
    # Remove BOM
    text = text.replace('\ufeff', '')
    
    # Normalize to NFC
    text = unicodedata.normalize('NFC', text)
    
    # Remove zero-width characters
    text = re.sub(r'[\u200b-\u200f\u202a-\u202e\ufeff]', '', text)
    
    return text


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace while preserving structure"""
    if not text:
        return ""
    
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Remove non-printable control characters (except tab and newline)
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    # Collapse multiple spaces (but keep single spaces)
    text = re.sub(r'[ ]{2,}', ' ', text)
    
    # Collapse multiple newlines to max 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def normalize_text(text: str) -> str:
    """Full text normalization pipeline"""
    if not text:
        return ""
    
    text = normalize_unicode(text)
    text = normalize_whitespace(text)
    
    return text


# ============================================================================
# TIMESTAMP PARSING
# ============================================================================

# Common timestamp patterns (order matters - most specific first)
TIMESTAMP_PATTERNS = [
    # ISO8601 / RFC3339
    (r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+\-]\d{2}:?\d{2})?', '%Y-%m-%dT%H:%M:%S'),
    (r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', '%Y-%m-%d %H:%M:%S'),
    
    # Syslog / rsyslog
    (r'[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}', '%b %d %H:%M:%S'),
    
    # Apache Common/Combined
    (r'\d{2}/[A-Z][a-z]{2}/\d{4}:\d{2}:\d{2}:\d{2}\s+[+\-]\d{4}', '%d/%b/%Y:%H:%M:%S %z'),
    
    # Nginx
    (r'\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2}', '%Y/%m/%d %H:%M:%S'),
    
    # Suricata / JSON timestamp
    (r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+[+\-]\d{2}:\d{2}', '%Y-%m-%dT%H:%M:%S.%f%z'),
    
    # Unix epoch (10 digits = seconds, 13 digits = milliseconds)
    (r'\b\d{10}\b', 'epoch_s'),
    (r'\b\d{13}\b', 'epoch_ms'),
]


def parse_timestamp(ts_str: str, current_year: int = None) -> Optional[datetime]:
    """Parse timestamp from various formats"""
    if not ts_str:
        return None
    
    ts_str = str(ts_str).strip()
    
    if current_year is None:
        current_year = datetime.now().year
    
    for pattern, fmt in TIMESTAMP_PATTERNS:
        match = re.search(pattern, ts_str)
        if match:
            matched_str = match.group(0)
            
            try:
                if fmt == 'epoch_s':
                    return datetime.fromtimestamp(int(matched_str), tz=timezone.utc)
                elif fmt == 'epoch_ms':
                    return datetime.fromtimestamp(int(matched_str) / 1000, tz=timezone.utc)
                else:
                    dt = datetime.strptime(matched_str, fmt)
                    
                    # Handle syslog format without year
                    if fmt == '%b %d %H:%M:%S':
                        dt = dt.replace(year=current_year)
                    
                    # Ensure UTC
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    else:
                        dt = dt.astimezone(timezone.utc)
                    
                    return dt
            except (ValueError, OSError):
                continue
    
    return None


def extract_timestamp_from_text(text: str) -> Optional[datetime]:
    """Try to extract timestamp from log text"""
    if not text:
        return None
    
    # Try first 200 chars where timestamp usually is
    prefix = text[:200]
    return parse_timestamp(prefix)


# ============================================================================
# STRUCTURED FIELD EXTRACTION
# ============================================================================

def extract_json_fields(text: str) -> Dict[str, Any]:
    """Extract fields from JSON log lines"""
    fields = {}
    
    # Try to parse as JSON
    text_stripped = text.strip()
    if text_stripped.startswith('{') and text_stripped.endswith('}'):
        try:
            data = json.loads(text_stripped)
            if isinstance(data, dict):
                # Flatten nested dicts with dot notation
                fields = flatten_dict(data)
        except json.JSONDecodeError:
            pass
    
    return fields


def flatten_dict(d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
    """Flatten nested dictionary with dot notation"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def extract_kv_fields(text: str) -> Dict[str, str]:
    """Extract key=value pairs from log text"""
    fields = {}
    
    # Patterns: key=value, key="value", key:[value]
    patterns = [
        r'(\w+)=(["\'])(.*?)\2',  # key="value" or key='value'
        r'(\w+)=([^\s,;]+)',       # key=value
        r'(\w+):\s*([^\s,;]+)',    # key: value
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            key = match.group(1)
            value = match.group(3) if len(match.groups()) > 2 else match.group(2)
            if key and value and len(key) <= 50:
                fields[key] = value
    
    return fields


def extract_common_entities(text: str) -> Dict[str, List[str]]:
    """Extract IPs, emails, UUIDs, etc."""
    entities = defaultdict(list)
    
    # IP addresses
    ips = re.findall(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', text)
    if ips:
        entities['ips'] = list(set(ips))[:10]  # limit to 10
    
    # Emails
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    if emails:
        entities['emails'] = list(set(emails))[:5]
    
    # UUIDs
    uuids = re.findall(r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b', text)
    if uuids:
        entities['uuids'] = list(set(uuids))[:5]
    
    # Hex digests (MD5, SHA256)
    hexes = re.findall(r'\b[0-9a-fA-F]{32,64}\b', text)
    if hexes:
        entities['hashes'] = list(set(hexes))[:5]
    
    # URLs
    urls = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', text)
    if urls:
        entities['urls'] = list(set(urls))[:5]
    
    return dict(entities)


def extract_structured_fields(text: str) -> Dict[str, Any]:
    """Extract all structured fields from log text"""
    structured = {}
    
    # Try JSON first
    json_fields = extract_json_fields(text)
    if json_fields:
        structured.update(json_fields)
    
    # Extract key=value pairs
    kv_fields = extract_kv_fields(text)
    if kv_fields:
        structured['kv'] = kv_fields
    
    # Extract entities
    entities = extract_common_entities(text)
    if entities:
        structured['entities'] = entities
    
    return structured


# ============================================================================
# REDACTION
# ============================================================================

def redact_sensitive_tokens(text: str) -> str:
    """Redact sensitive information with stable placeholders"""
    if not text:
        return text
    
    # IP addresses
    text = re.sub(
        r'\b((?:[0-9]{1,3}\.){3}[0-9]{1,3})\b',
        lambda m: f"<IP:{stable_hash(m.group(1))}>",
        text
    )
    
    # Email addresses
    text = re.sub(
        r'\b([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})\b',
        lambda m: f"<EMAIL:{stable_hash(m.group(1))}>",
        text
    )
    
    # MAC addresses
    text = re.sub(
        r'\b([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})\b',
        lambda m: f"<MAC:{stable_hash(m.group(0))}>",
        text
    )
    
    # UUIDs
    text = re.sub(
        r'\b([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})\b',
        lambda m: f"<UUID:{stable_hash(m.group(1))}>",
        text
    )
    
    # Credit card patterns (simple)
    text = re.sub(
        r'\b(\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4})\b',
        lambda m: f"<CC:{stable_hash(m.group(1))}>",
        text
    )
    
    return text


# ============================================================================
# CANONICALIZATION (for template mining)
# ============================================================================

CANONICALIZATION_PATTERNS = [
    # Timestamps (various formats)
    (re.compile(r'\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+\-]\d{2}:?\d{2})?\b'), '<TS>'),
    (re.compile(r'\b\d{2}/[A-Z][a-z]{2}/\d{4}:\d{2}:\d{2}:\d{2}\b'), '<TS>'),
    (re.compile(r'\b[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\b'), '<TS>'),
    
    # IPs
    (re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'), '<IP>'),
    
    # Hex hashes
    (re.compile(r'\b[0-9a-fA-F]{32,64}\b'), '<HEX>'),
    
    # UUIDs
    (re.compile(r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b'), '<UUID>'),
    
    # Quoted strings
    (re.compile(r'"[^"]*"'), '<STR>'),
    (re.compile(r"'[^']*'"), '<STR>'),
    
    # PIDs
    (re.compile(r'\[(\d+)\]'), '[<PID>]'),
    
    # File paths
    (re.compile(r'(?:/[a-zA-Z0-9_\-\.]+){2,}'), '<PATH>'),
    
    # Numbers (do last)
    (re.compile(r'\b\d+\b'), '<NUM>'),
]


def canonicalize_for_templates(text: str) -> str:
    """Create canonical version for template extraction"""
    if not text:
        return ""
    
    result = text
    for pattern, replacement in CANONICALIZATION_PATTERNS:
        result = pattern.sub(replacement, result)
    
    # Collapse whitespace
    result = re.sub(r'\s+', ' ', result).strip()
    
    return result


# ============================================================================
# HOST AND LOG_TYPE SANITIZATION
# ============================================================================

def sanitize_identifier(value: str) -> str:
    """Sanitize host or log_type value"""
    if not value:
        return "unknown"
    
    value = str(value).strip().lower()
    value = re.sub(r'\s+', '_', value)
    value = re.sub(r'[^\w\-\.]', '_', value)
    value = re.sub(r'_+', '_', value).strip('_')
    
    return value if value else "unknown"


def canonicalize_log_type(log_type: str) -> str:
    """Map log_type to canonical name"""
    log_type = sanitize_identifier(log_type)
    return CleaningConfig.LOG_TYPE_CANONICAL.get(log_type, log_type)


def get_coarse_log_type(log_type: str) -> str:
    """Get coarse category for log_type"""
    return CleaningConfig.COARSE_LOG_TYPE.get(log_type, "other")


# ============================================================================
# SEVERITY EXTRACTION
# ============================================================================

SEVERITY_PATTERNS = [
    (re.compile(r'\b(EMERG|EMERGENCY)\b', re.IGNORECASE), 'EMERGENCY'),
    (re.compile(r'\b(ALERT)\b', re.IGNORECASE), 'ALERT'),
    (re.compile(r'\b(CRIT|CRITICAL)\b', re.IGNORECASE), 'CRITICAL'),
    (re.compile(r'\b(ERR|ERROR)\b', re.IGNORECASE), 'ERROR'),
    (re.compile(r'\b(WARN|WARNING)\b', re.IGNORECASE), 'WARNING'),
    (re.compile(r'\b(NOTICE)\b', re.IGNORECASE), 'NOTICE'),
    (re.compile(r'\b(INFO|INFORMATION)\b', re.IGNORECASE), 'INFO'),
    (re.compile(r'\b(DEBUG)\b', re.IGNORECASE), 'DEBUG'),
]


def extract_severity(text: str) -> Optional[str]:
    """Extract severity level from log text"""
    if not text:
        return None
    
    # Check first 100 chars where severity usually appears
    prefix = text[:100]
    
    for pattern, level in SEVERITY_PATTERNS:
        if pattern.search(prefix):
            return level
    
    return None


# ============================================================================
# MAIN PROCESSING PIPELINE
# ============================================================================

def process_batch(batch: Dict[str, List]) -> Dict[str, List]:
    """Process a batch of log records"""
    
    # Initialize output columns
    output = {
        'row_id': [],
        'timestamp_raw': [],
        'timestamp': [],
        'epoch_ms': [],
        'host': [],
        'host_sanitized': [],
        'log_type': [],
        'log_type_canonical': [],
        'coarse_log_type': [],
        'text_original': [],
        'text': [],
        'text_canonical': [],
        'text_len': [],
        'text_was_truncated': [],
        'structured_fields': [],
        'severity': [],
        'label': [],
        'is_attack': [],
        'hour_of_day': [],
        'day_of_week': [],
        'is_weekend': [],
        'rounded_ts_1m': [],
        'rounded_ts_5m': [],
        'rounded_ts_30m': [],
    }
    
    batch_size = len(batch[list(batch.keys())[0]])
    
    for i in range(batch_size):
        # Extract fields from batch
        row_data = {k: v[i] for k, v in batch.items()}
        
        # Generate row ID
        row_id = f"row_{i}_{hashlib.md5(str(row_data).encode()).hexdigest()[:8]}"
        
        # Get raw values
        host_raw = row_data.get('host', '')
        log_type_raw = row_data.get('log_type', '')
        timestamp_raw = row_data.get('timestamp', '')
        text_raw = row_data.get('content', '') or row_data.get('text', '') or row_data.get('message', '')
        label_raw = row_data.get('label', '')
        is_attack_raw = row_data.get('is_attack', False)
        
        # Sanitize host and log_type
        host = sanitize_identifier(host_raw)
        log_type = sanitize_identifier(log_type_raw)
        log_type_canonical = canonicalize_log_type(log_type)
        coarse_log_type = get_coarse_log_type(log_type_canonical)
        
        # Normalize text
        text_original = str(text_raw) if text_raw else ""
        text = normalize_text(text_original)
        
        # Truncate if needed
        text_was_truncated = len(text) > CleaningConfig.MAX_TEXT_LENGTH
        if text_was_truncated:
            text = text[:CleaningConfig.MAX_TEXT_LENGTH]
        
        text_len = len(text)
        
        # Parse timestamp
        ts = parse_timestamp(timestamp_raw)
        if ts is None:
            ts = extract_timestamp_from_text(text)
        
        timestamp_iso = ts.isoformat() if ts else None
        epoch_ms = int(ts.timestamp() * 1000) if ts else None
        
        # Extract structured fields
        structured = extract_structured_fields(text)
        
        # Redact sensitive info
        text_redacted = redact_sensitive_tokens(text)
        
        # Canonicalize for templates
        text_canonical = canonicalize_for_templates(text_redacted)
        
        # Extract severity
        severity = extract_severity(text)
        
        # Temporal features
        hour_of_day = ts.hour if ts else None
        day_of_week = ts.weekday() if ts else None  # 0=Monday, 6=Sunday
        is_weekend = (day_of_week >= 5) if day_of_week is not None else None
        
        # Rounded timestamps for sessionization
        rounded_ts_1m = ts.replace(second=0, microsecond=0).isoformat() if ts else None
        rounded_ts_5m = ts.replace(minute=(ts.minute // 5) * 5, second=0, microsecond=0).isoformat() if ts else None
        rounded_ts_30m = ts.replace(minute=(ts.minute // 30) * 30, second=0, microsecond=0).isoformat() if ts else None
        
        # Normalize label
        label = None
        if label_raw:
            label_lower = str(label_raw).lower()
            if 'normal' in label_lower or label_lower == '0':
                label = 'normal'
            elif 'anomaly' in label_lower or 'attack' in label_lower or label_lower == '1':
                label = 'anomaly'
        
        is_attack = bool(is_attack_raw)
        
        # Append to output
        output['row_id'].append(row_id)
        output['timestamp_raw'].append(str(timestamp_raw) if timestamp_raw else None)
        output['timestamp'].append(timestamp_iso)
        output['epoch_ms'].append(epoch_ms)
        output['host'].append(host_raw)
        output['host_sanitized'].append(host)
        output['log_type'].append(log_type_raw)
        output['log_type_canonical'].append(log_type_canonical)
        output['coarse_log_type'].append(coarse_log_type)
        output['text_original'].append(text_original)
        output['text'].append(text_redacted)
        output['text_canonical'].append(text_canonical)
        output['text_len'].append(text_len)
        output['text_was_truncated'].append(text_was_truncated)
        output['structured_fields'].append(json.dumps(structured) if structured else None)
        output['severity'].append(severity)
        output['label'].append(label)
        output['is_attack'].append(is_attack)
        output['hour_of_day'].append(hour_of_day)
        output['day_of_week'].append(day_of_week)
        output['is_weekend'].append(is_weekend)
        output['rounded_ts_1m'].append(rounded_ts_1m)
        output['rounded_ts_5m'].append(rounded_ts_5m)
        output['rounded_ts_30m'].append(rounded_ts_30m)
    
    return output


def filter_garbage(dataset: Dataset) -> Dataset:
    """Filter out garbage/invalid rows"""
    
    def is_valid(example):
        # Must have minimum text length
        if not example['text'] or len(example['text']) < CleaningConfig.MIN_TEXT_LENGTH:
            return False
        
        # Must have timestamp
        if not example['timestamp']:
            return False
        
        # Check if mostly binary/noise
        text = example['text']
        alnum_count = sum(c.isalnum() for c in text[:200])
        if alnum_count < len(text[:200]) * 0.3:  # Less than 30% alphanumeric
            return False
        
        return True
    
    return dataset.filter(is_valid, desc="Filtering garbage")


def deduplicate_exact(dataset: Dataset) -> Dataset:
    """Remove exact duplicates within time windows"""
    
    # Create dedup key
    def add_dedup_key(example):
        # Round timestamp to nearest N seconds
        epoch = example['epoch_ms']
        if epoch:
            epoch_bucket = (epoch // (CleaningConfig.DEDUP_TIME_WINDOW_SECONDS * 1000)) * (CleaningConfig.DEDUP_TIME_WINDOW_SECONDS * 1000)
        else:
            epoch_bucket = 0
        
        # Create composite key
        text_short = example['text'][:500] if example['text'] else ""
        dedup_key = f"{example['host_sanitized']}:{example['log_type_canonical']}:{epoch_bucket}:{hashlib.md5(text_short.encode()).hexdigest()}"
        return {'dedup_key': dedup_key}
    
    dataset = dataset.map(add_dedup_key, desc="Creating dedup keys")
    
    # Keep track of seen keys
    seen = set()
    
    def is_not_duplicate(example):
        key = example['dedup_key']
        if key in seen:
            return False
        seen.add(key)
        return True
    
    dataset = dataset.filter(is_not_duplicate, desc="Removing exact duplicates")
    
    # Remove dedup_key column
    dataset = dataset.remove_columns(['dedup_key'])
    
    return dataset


# ============================================================================
# STATISTICS AND REPORTING
# ============================================================================

def compute_statistics(dataset: Dataset) -> Dict:
    """Compute dataset statistics"""
    
    stats = {
        'total_rows': len(dataset),
        'host_counts': Counter(),
        'log_type_counts': Counter(),
        'coarse_type_counts': Counter(),
        'severity_counts': Counter(),
        'label_counts': Counter(),
        'timestamp_parse_rate': 0,
        'avg_text_length': 0,
        'text_length_percentiles': {},
        'truncation_rate': 0,
    }
    
    valid_ts_count = 0
    text_lengths = []
    truncated_count = 0
    
    for row in tqdm(dataset, desc="Computing statistics"):
        stats['host_counts'][row['host_sanitized']] += 1
        stats['log_type_counts'][row['log_type_canonical']] += 1
        stats['coarse_type_counts'][row['coarse_log_type']] += 1
        
        if row['severity']:
            stats['severity_counts'][row['severity']] += 1
        
        if row['label']:
            stats['label_counts'][row['label']] += 1
        
        if row['timestamp']:
            valid_ts_count += 1
        
        if row['text_len']:
            text_lengths.append(row['text_len'])
        
        if row['text_was_truncated']:
            truncated_count += 1
    
    stats['timestamp_parse_rate'] = valid_ts_count / len(dataset) if len(dataset) > 0 else 0
    stats['avg_text_length'] = sum(text_lengths) / len(text_lengths) if text_lengths else 0
    stats['truncation_rate'] = truncated_count / len(dataset) if len(dataset) > 0 else 0
    
    if text_lengths:
        sorted_lengths = sorted(text_lengths)
        stats['text_length_percentiles'] = {
            'p50': sorted_lengths[len(sorted_lengths) // 2],
            'p90': sorted_lengths[int(len(sorted_lengths) * 0.9)],
            'p95': sorted_lengths[int(len(sorted_lengths) * 0.95)],
            'p99': sorted_lengths[int(len(sorted_lengths) * 0.99)],
        }
    
    # Convert counters to dicts
    stats['host_counts'] = dict(stats['host_counts'].most_common(50))
    stats['log_type_counts'] = dict(stats['log_type_counts'].most_common(50))
    stats['coarse_type_counts'] = dict(stats['coarse_type_counts'])
    stats['severity_counts'] = dict(stats['severity_counts'])
    stats['label_counts'] = dict(stats['label_counts'])
    
    return stats


def write_quality_report(stats: Dict, output_path: str):
    """Write quality report markdown"""
    
    report = f"""# Log Cleaning Quality Report

Generated: {datetime.now().isoformat()}

## Dataset Overview

- **Total Rows**: {stats['total_rows']:,}
- **Timestamp Parse Rate**: {stats['timestamp_parse_rate']:.2%}
- **Average Text Length**: {stats['avg_text_length']:.0f} chars
- **Truncation Rate**: {stats['truncation_rate']:.2%}

## Text Length Distribution

- **P50**: {stats['text_length_percentiles'].get('p50', 0)} chars
- **P90**: {stats['text_length_percentiles'].get('p90', 0)} chars
- **P95**: {stats['text_length_percentiles'].get('p95', 0)} chars
- **P99**: {stats['text_length_percentiles'].get('p99', 0)} chars

## Top Hosts

| Host | Count |
|------|-------|
"""
    
    for host, count in list(stats['host_counts'].items())[:20]:
        report += f"| {host} | {count:,} |\n"
    
    report += "\n## Log Types\n\n| Log Type | Count |\n|----------|-------|\n"
    
    for log_type, count in list(stats['log_type_counts'].items())[:20]:
        report += f"| {log_type} | {count:,} |\n"
    
    report += "\n## Coarse Categories\n\n| Category | Count |\n|----------|-------|\n"
    
    for cat, count in stats['coarse_type_counts'].items():
        report += f"| {cat} | {count:,} |\n"
    
    if stats['severity_counts']:
        report += "\n## Severity Distribution\n\n| Severity | Count |\n|----------|-------|\n"
        for sev, count in stats['severity_counts'].items():
            report += f"| {sev} | {count:,} |\n"
    
    if stats['label_counts']:
        report += "\n## Label Distribution\n\n| Label | Count |\n|-------|-------|\n"
        for label, count in stats['label_counts'].items():
            report += f"| {label} | {count:,} |\n"
    
    with open(output_path, 'w') as f:
        f.write(report)


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Clean HF logs for training")
    parser.add_argument("--hf_repo", type=str, default="chYassine/ait-fox-raw-v02", help="HF repo name")
    parser.add_argument("--out_dir", type=str, default=CleaningConfig.OUT_DIR, help="Output directory")
    parser.add_argument("--push_hf", type=str, default="false", help="Push to HF hub")
    parser.add_argument("--hf_out_repo", type=str, default="", help="Output HF repo name")
    
    args = parser.parse_args()
    
    # Update config
    CleaningConfig.OUT_DIR = args.out_dir
    ensure_dirs()
    
    # Initialize salt
    global REDACTION_SALT
    REDACTION_SALT = get_redaction_salt()
    
    print(f"Loading dataset from {args.hf_repo}...")
    dataset = load_dataset(args.hf_repo, split="train")
    
    print(f"Loaded {len(dataset):,} rows")
    
    # Process in batches
    print("Processing records...")
    dataset = dataset.map(
        process_batch,
        batched=True,
        batch_size=1000,
        remove_columns=dataset.column_names,
        desc="Processing logs"
    )
    
    # Filter garbage
    print("Filtering invalid rows...")
    initial_count = len(dataset)
    dataset = filter_garbage(dataset)
    filtered_count = initial_count - len(dataset)
    print(f"  Removed {filtered_count:,} invalid rows ({filtered_count/initial_count:.1%})")
    
    # Deduplicate
    print("Deduplicating...")
    initial_count = len(dataset)
    dataset = deduplicate_exact(dataset)
    dedup_count = initial_count - len(dataset)
    print(f"  Removed {dedup_count:,} duplicate rows ({dedup_count/initial_count:.1%})")
    
    # Compute statistics
    print("Computing statistics...")
    stats = compute_statistics(dataset)
    
    # Write outputs
    print("Saving outputs...")
    
    # Save dataset
    dataset.save_to_disk(CleaningConfig.OUT_DIR)
    print(f"  Dataset saved to {CleaningConfig.OUT_DIR}")
    
    # Save stats
    stats_path = f"{CleaningConfig.ARTIFACTS_DIR}/stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats saved to {stats_path}")
    
    # Write quality report
    report_path = f"{CleaningConfig.ARTIFACTS_DIR}/quality_report.md"
    write_quality_report(stats, report_path)
    print(f"  Report saved to {report_path}")
    
    # Save sample
    sample = dataset.shuffle(seed=42).select(range(min(100, len(dataset))))
    sample_df = sample.to_pandas()
    sample_path = f"{CleaningConfig.ARTIFACTS_DIR}/samples/sample_train.csv"
    sample_df.to_csv(sample_path, index=False)
    print(f"  Sample saved to {sample_path}")
    
    # Push to HF if requested
    if args.push_hf.lower() == "true" and args.hf_out_repo:
        print(f"Pushing to HuggingFace: {args.hf_out_repo}...")
        dataset.push_to_hub(args.hf_out_repo)
        print("  ✓ Pushed to HF")
    
    print("\n✅ Cleaning pipeline complete!")
    print(f"  Final row count: {len(dataset):,}")
    print(f"  Timestamp parse rate: {stats['timestamp_parse_rate']:.1%}")
    print(f"  Average text length: {stats['avg_text_length']:.0f} chars")


if __name__ == "__main__":
    main()

