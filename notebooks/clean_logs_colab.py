# LOG CLEANING PIPELINE - COLAB VERSION
# Run this cell-by-cell in Google Colab

# ============================================================================
# CELL 1: Install dependencies
# ============================================================================
"""
!pip -q install datasets pandas tqdm
"""

# ============================================================================
# CELL 2: Configuration and imports
# ============================================================================

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
from datasets import load_dataset, Dataset
from tqdm import tqdm

# Configuration
HF_REPO = "chYassine/ait-fox-raw-v02"
OUT_DIR = "datasets/ait/output/ait/cleaned"
ARTIFACTS_DIR = "artifacts/cleaning"
MAX_TEXT_LENGTH = 8192
MIN_TEXT_LENGTH = 3
DEDUP_TIME_WINDOW_SECONDS = 2

# Create directories
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(f"{ARTIFACTS_DIR}/samples", exist_ok=True)

# Initialize redaction salt
REDACTION_SALT = hashlib.sha256(str(datetime.now().timestamp()).encode()).hexdigest()[:16]
print(f"✓ Configuration loaded")
print(f"  Repository: {HF_REPO}")
print(f"  Output: {OUT_DIR}")

# ============================================================================
# CELL 3: Utility functions
# ============================================================================

def stable_hash(value: str, prefix: str = "") -> str:
    """Create stable hash for redaction"""
    salted = f"{REDACTION_SALT}:{value}"
    hash_val = hashlib.sha1(salted.encode()).hexdigest()[:8]
    return f"{prefix}{hash_val}" if prefix else hash_val


def normalize_unicode(text: str) -> str:
    """Normalize unicode to NFC form"""
    if not text:
        return ""
    text = text.replace('\ufeff', '')
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'[\u200b-\u200f\u202a-\u202e\ufeff]', '', text)
    return text


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace"""
    if not text:
        return ""
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    text = re.sub(r'[ ]{2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def normalize_text(text: str) -> str:
    """Full text normalization"""
    if not text:
        return ""
    text = normalize_unicode(text)
    text = normalize_whitespace(text)
    return text

print("✓ Utility functions loaded")

# ============================================================================
# CELL 4: Timestamp parsing
# ============================================================================

TIMESTAMP_PATTERNS = [
    (r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+\-]\d{2}:?\d{2})?', '%Y-%m-%dT%H:%M:%S'),
    (r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', '%Y-%m-%d %H:%M:%S'),
    (r'[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}', '%b %d %H:%M:%S'),
    (r'\d{2}/[A-Z][a-z]{2}/\d{4}:\d{2}:\d{2}:\d{2}\s+[+\-]\d{4}', '%d/%b/%Y:%H:%M:%S %z'),
    (r'\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2}', '%Y/%m/%d %H:%M:%S'),
    (r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+[+\-]\d{2}:\d{2}', '%Y-%m-%dT%H:%M:%S.%f%z'),
    (r'\b\d{10}\b', 'epoch_s'),
    (r'\b\d{13}\b', 'epoch_ms'),
]


def parse_timestamp(ts_str: str) -> Optional[datetime]:
    """Parse timestamp from various formats"""
    if not ts_str:
        return None
    
    ts_str = str(ts_str).strip()
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
                    if fmt == '%b %d %H:%M:%S':
                        dt = dt.replace(year=current_year)
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
    prefix = text[:200]
    return parse_timestamp(prefix)

print("✓ Timestamp parsing loaded")

# ============================================================================
# CELL 5: Structured field extraction
# ============================================================================

def flatten_dict(d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
    """Flatten nested dictionary"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def extract_json_fields(text: str) -> Dict[str, Any]:
    """Extract fields from JSON log lines"""
    fields = {}
    text_stripped = text.strip()
    if text_stripped.startswith('{') and text_stripped.endswith('}'):
        try:
            data = json.loads(text_stripped)
            if isinstance(data, dict):
                fields = flatten_dict(data)
        except json.JSONDecodeError:
            pass
    return fields


def extract_kv_fields(text: str) -> Dict[str, str]:
    """Extract key=value pairs"""
    fields = {}
    patterns = [
        r'(\w+)=(["\'])(.*?)\2',
        r'(\w+)=([^\s,;]+)',
        r'(\w+):\s*([^\s,;]+)',
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
    
    ips = re.findall(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', text)
    if ips:
        entities['ips'] = list(set(ips))[:10]
    
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    if emails:
        entities['emails'] = list(set(emails))[:5]
    
    uuids = re.findall(r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b', text)
    if uuids:
        entities['uuids'] = list(set(uuids))[:5]
    
    return dict(entities)


def extract_structured_fields(text: str) -> Dict[str, Any]:
    """Extract all structured fields"""
    structured = {}
    
    json_fields = extract_json_fields(text)
    if json_fields:
        structured.update(json_fields)
    
    kv_fields = extract_kv_fields(text)
    if kv_fields:
        structured['kv'] = kv_fields
    
    entities = extract_common_entities(text)
    if entities:
        structured['entities'] = entities
    
    return structured

print("✓ Structured extraction loaded")

# ============================================================================
# CELL 6: Redaction and canonicalization
# ============================================================================

def redact_sensitive_tokens(text: str) -> str:
    """Redact sensitive information"""
    if not text:
        return text
    
    text = re.sub(
        r'\b((?:[0-9]{1,3}\.){3}[0-9]{1,3})\b',
        lambda m: f"<IP:{stable_hash(m.group(1))}>",
        text
    )
    text = re.sub(
        r'\b([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})\b',
        lambda m: f"<EMAIL:{stable_hash(m.group(1))}>",
        text
    )
    text = re.sub(
        r'\b([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})\b',
        lambda m: f"<MAC:{stable_hash(m.group(0))}>",
        text
    )
    text = re.sub(
        r'\b([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})\b',
        lambda m: f"<UUID:{stable_hash(m.group(1))}>",
        text
    )
    
    return text


CANONICALIZATION_PATTERNS = [
    (re.compile(r'\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+\-]\d{2}:?\d{2})?\b'), '<TS>'),
    (re.compile(r'\b\d{2}/[A-Z][a-z]{2}/\d{4}:\d{2}:\d{2}:\d{2}\b'), '<TS>'),
    (re.compile(r'\b[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\b'), '<TS>'),
    (re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'), '<IP>'),
    (re.compile(r'\b[0-9a-fA-F]{32,64}\b'), '<HEX>'),
    (re.compile(r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b'), '<UUID>'),
    (re.compile(r'"[^"]*"'), '<STR>'),
    (re.compile(r"'[^']*'"), '<STR>'),
    (re.compile(r'\[(\d+)\]'), '[<PID>]'),
    (re.compile(r'(?:/[a-zA-Z0-9_\-\.]+){2,}'), '<PATH>'),
    (re.compile(r'\b\d+\b'), '<NUM>'),
]


def canonicalize_for_templates(text: str) -> str:
    """Create canonical version for template extraction"""
    if not text:
        return ""
    result = text
    for pattern, replacement in CANONICALIZATION_PATTERNS:
        result = pattern.sub(replacement, result)
    result = re.sub(r'\s+', ' ', result).strip()
    return result

print("✓ Redaction and canonicalization loaded")

# ============================================================================
# CELL 7: Host/log_type sanitization
# ============================================================================

LOG_TYPE_CANONICAL = {
    "apache": "apache2", "apache_access": "apache2", "httpd": "apache2",
    "nginx": "nginx", "web": "apache2",
    "suricata": "suricata", "ids": "suricata", "nids": "suricata",
    "auth": "auth", "sshd": "auth", "pam": "auth",
    "syslog": "journal", "system": "journal", "systemd": "journal",
    "exim": "exim4", "mail": "exim4", "postfix": "postfix",
    "dns": "dns", "bind": "dns",
    "auditd": "audit",
    "logstash": "logstash", "elasticsearch": "logstash",
    "horde": "horde",
}

COARSE_LOG_TYPE = {
    "apache2": "web", "nginx": "web",
    "suricata": "network",
    "auth": "auth",
    "journal": "system",
    "exim4": "mail", "postfix": "mail", "horde": "mail",
    "dns": "network",
    "audit": "system",
    "logstash": "monitoring",
    "dnsteal": "network",
    "ait.aecid.attacker.wpdiscuz": "attack",
    "downloads": "application",
    "redis": "database",
}


def sanitize_identifier(value: str) -> str:
    """Sanitize host or log_type"""
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
    return LOG_TYPE_CANONICAL.get(log_type, log_type)


def get_coarse_log_type(log_type: str) -> str:
    """Get coarse category"""
    return COARSE_LOG_TYPE.get(log_type, "other")

print("✓ Sanitization functions loaded")

# ============================================================================
# CELL 8: Severity extraction
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
    """Extract severity level"""
    if not text:
        return None
    prefix = text[:100]
    for pattern, level in SEVERITY_PATTERNS:
        if pattern.search(prefix):
            return level
    return None

print("✓ Severity extraction loaded")

# ============================================================================
# CELL 9: Main processing function
# ============================================================================

def process_batch(batch: Dict[str, List]) -> Dict[str, List]:
    """Process a batch of log records"""
    
    output = {
        'row_id': [], 'timestamp_raw': [], 'timestamp': [], 'epoch_ms': [],
        'host': [], 'host_sanitized': [], 'log_type': [], 'log_type_canonical': [],
        'coarse_log_type': [], 'text_original': [], 'text': [], 'text_canonical': [],
        'text_len': [], 'text_was_truncated': [], 'structured_fields': [],
        'severity': [], 'label': [], 'is_attack': [],
        'hour_of_day': [], 'day_of_week': [], 'is_weekend': [],
        'rounded_ts_1m': [], 'rounded_ts_5m': [], 'rounded_ts_30m': [],
    }
    
    batch_size = len(batch[list(batch.keys())[0]])
    
    for i in range(batch_size):
        row_data = {k: v[i] for k, v in batch.items()}
        
        row_id = f"row_{i}_{hashlib.md5(str(row_data).encode()).hexdigest()[:8]}"
        
        host_raw = row_data.get('host', '')
        log_type_raw = row_data.get('log_type', '')
        timestamp_raw = row_data.get('timestamp', '')
        text_raw = row_data.get('content', '') or row_data.get('text', '') or row_data.get('message', '')
        label_raw = row_data.get('label', '')
        is_attack_raw = row_data.get('is_attack', False)
        
        host = sanitize_identifier(host_raw)
        log_type = sanitize_identifier(log_type_raw)
        log_type_canonical = canonicalize_log_type(log_type)
        coarse_log_type = get_coarse_log_type(log_type_canonical)
        
        text_original = str(text_raw) if text_raw else ""
        text = normalize_text(text_original)
        
        text_was_truncated = len(text) > MAX_TEXT_LENGTH
        if text_was_truncated:
            text = text[:MAX_TEXT_LENGTH]
        
        text_len = len(text)
        
        ts = parse_timestamp(timestamp_raw)
        if ts is None:
            ts = extract_timestamp_from_text(text)
        
        timestamp_iso = ts.isoformat() if ts else ""
        epoch_ms = int(ts.timestamp() * 1000) if ts else 0
        
        structured = extract_structured_fields(text)
        text_redacted = redact_sensitive_tokens(text)
        text_canonical = canonicalize_for_templates(text_redacted)
        severity = extract_severity(text)
        
        hour_of_day = ts.hour if ts else -1
        day_of_week = ts.weekday() if ts else -1
        is_weekend = (day_of_week >= 5) if day_of_week >= 0 else False
        
        rounded_ts_1m = ts.replace(second=0, microsecond=0).isoformat() if ts else ""
        rounded_ts_5m = ts.replace(minute=(ts.minute // 5) * 5, second=0, microsecond=0).isoformat() if ts else ""
        rounded_ts_30m = ts.replace(minute=(ts.minute // 30) * 30, second=0, microsecond=0).isoformat() if ts else ""
        
        label = ""
        if label_raw:
            label_lower = str(label_raw).lower()
            if 'normal' in label_lower or label_lower == '0':
                label = 'normal'
            elif 'anomaly' in label_lower or 'attack' in label_lower or label_lower == '1':
                label = 'anomaly'
        
        is_attack = bool(is_attack_raw)
        
        # Use empty strings instead of None for string fields to avoid type casting issues
        output['row_id'].append(row_id)
        output['timestamp_raw'].append(str(timestamp_raw) if timestamp_raw else "")
        output['timestamp'].append(timestamp_iso)
        output['epoch_ms'].append(epoch_ms)
        output['host'].append(str(host_raw) if host_raw else "")
        output['host_sanitized'].append(host)
        output['log_type'].append(str(log_type_raw) if log_type_raw else "")
        output['log_type_canonical'].append(log_type_canonical)
        output['coarse_log_type'].append(coarse_log_type)
        output['text_original'].append(text_original)
        output['text'].append(text_redacted)
        output['text_canonical'].append(text_canonical)
        output['text_len'].append(text_len)
        output['text_was_truncated'].append(text_was_truncated)
        output['structured_fields'].append(json.dumps(structured) if structured else "")
        output['severity'].append(severity if severity else "")
        output['label'].append(label)
        output['is_attack'].append(is_attack)
        output['hour_of_day'].append(hour_of_day)
        output['day_of_week'].append(day_of_week)
        output['is_weekend'].append(is_weekend)
        output['rounded_ts_1m'].append(rounded_ts_1m)
        output['rounded_ts_5m'].append(rounded_ts_5m)
        output['rounded_ts_30m'].append(rounded_ts_30m)
    
    return output

print("✓ Main processing function loaded")

# ============================================================================
# CELL 10: Filtering and deduplication
# ============================================================================

def filter_garbage(dataset: Dataset) -> Dataset:
    """Filter out garbage/invalid rows"""
    def is_valid(example):
        # Must have minimum text length
        if not example['text'] or len(example['text']) < MIN_TEXT_LENGTH:
            return False
        # Must have valid timestamp (not empty string)
        if not example['timestamp'] or example['timestamp'] == "":
            return False
        # Check if mostly binary/noise
        text = example['text']
        alnum_count = sum(c.isalnum() for c in text[:200])
        if alnum_count < len(text[:200]) * 0.3:
            return False
        return True
    
    return dataset.filter(is_valid, desc="Filtering garbage")


def deduplicate_exact(dataset: Dataset) -> Dataset:
    """Remove exact duplicates"""
    def add_dedup_key(example):
        epoch = example['epoch_ms']
        if epoch and epoch > 0:
            epoch_bucket = (epoch // (DEDUP_TIME_WINDOW_SECONDS * 1000)) * (DEDUP_TIME_WINDOW_SECONDS * 1000)
        else:
            epoch_bucket = 0
        text_short = example['text'][:500] if example['text'] else ""
        dedup_key = f"{example['host_sanitized']}:{example['log_type_canonical']}:{epoch_bucket}:{hashlib.md5(text_short.encode()).hexdigest()}"
        return {'dedup_key': dedup_key}
    
    dataset = dataset.map(add_dedup_key, desc="Creating dedup keys")
    
    seen = set()
    def is_not_duplicate(example):
        key = example['dedup_key']
        if key in seen:
            return False
        seen.add(key)
        return True
    
    dataset = dataset.filter(is_not_duplicate, desc="Removing exact duplicates")
    dataset = dataset.remove_columns(['dedup_key'])
    
    return dataset

print("✓ Filtering functions loaded")

# ============================================================================
# CELL 11: Statistics and reporting
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
    
    stats['host_counts'] = dict(stats['host_counts'].most_common(50))
    stats['log_type_counts'] = dict(stats['log_type_counts'].most_common(50))
    stats['coarse_type_counts'] = dict(stats['coarse_type_counts'])
    stats['severity_counts'] = dict(stats['severity_counts'])
    stats['label_counts'] = dict(stats['label_counts'])
    
    return stats

print("✓ Statistics functions loaded")

# ============================================================================
# CELL 12: Run the pipeline
# ============================================================================

print("="*60)
print("STARTING LOG CLEANING PIPELINE")
print("="*60)

# Load dataset
print(f"\n[1/7] Loading dataset from {HF_REPO}...")
dataset = load_dataset(HF_REPO, split="train")
print(f"  Loaded {len(dataset):,} rows")

# Process
print("\n[2/7] Processing records...")
dataset = dataset.map(
    process_batch,
    batched=True,
    batch_size=1000,
    remove_columns=dataset.column_names,
    desc="Processing logs"
)
print(f"  ✓ Processed {len(dataset):,} rows")

# Filter
print("\n[3/7] Filtering invalid rows...")
initial_count = len(dataset)
dataset = filter_garbage(dataset)
filtered_count = initial_count - len(dataset)
print(f"  ✓ Removed {filtered_count:,} invalid rows ({filtered_count/initial_count:.1%})")

# Deduplicate
print("\n[4/7] Deduplicating...")
initial_count = len(dataset)
dataset = deduplicate_exact(dataset)
dedup_count = initial_count - len(dataset)
print(f"  ✓ Removed {dedup_count:,} duplicate rows ({dedup_count/initial_count:.1%})")

# Statistics
print("\n[5/7] Computing statistics...")
stats = compute_statistics(dataset)

# Save
print("\n[6/7] Saving outputs...")
dataset.save_to_disk(OUT_DIR)
print(f"  ✓ Dataset saved to {OUT_DIR}")

with open(f"{ARTIFACTS_DIR}/stats.json", 'w') as f:
    json.dump(stats, f, indent=2)
print(f"  ✓ Stats saved")

# Sample
sample = dataset.shuffle(seed=42).select(range(min(100, len(dataset))))
sample_df = sample.to_pandas()
sample_df.to_csv(f"{ARTIFACTS_DIR}/samples/sample_train.csv", index=False)
print(f"  ✓ Sample saved")

# Report
print("\n[7/7] Final Statistics:")
print(f"  Total rows: {stats['total_rows']:,}")
print(f"  Timestamp parse rate: {stats['timestamp_parse_rate']:.1%}")
print(f"  Average text length: {stats['avg_text_length']:.0f} chars")
print(f"  Top 5 hosts: {list(stats['host_counts'].keys())[:5]}")
print(f"  Top 5 log types: {list(stats['log_type_counts'].keys())[:5]}")

print("\n" + "="*60)
print("✅ CLEANING PIPELINE COMPLETE!")
print("="*60)

# Display sample rows
print("\nSample cleaned rows:")
print(sample_df[['host_sanitized', 'log_type_canonical', 'timestamp', 'text_len', 'severity']].head(10))

