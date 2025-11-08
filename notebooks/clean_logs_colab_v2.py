# LOG CLEANING PIPELINE - COLAB VERSION V2 (Lenient Timestamp Filtering)
# This version is more lenient with timestamp filtering for logs where timestamp is embedded in content

# ============================================================================
# CELL 2: Configuration (same as before)
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

HF_REPO = "chYassine/ait-fox-raw-v02"
OUT_DIR = "datasets/ait/output/ait/cleaned"
ARTIFACTS_DIR = "artifacts/cleaning"
MAX_TEXT_LENGTH = 8192
MIN_TEXT_LENGTH = 3
DEDUP_TIME_WINDOW_SECONDS = 2

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(f"{ARTIFACTS_DIR}/samples", exist_ok=True)

REDACTION_SALT = hashlib.sha256(str(datetime.now().timestamp()).encode()).hexdigest()[:16]
print(f"✓ Configuration loaded (V2 - Lenient timestamp filtering)")
print(f"  Repository: {HF_REPO}")
print(f"  Output: {OUT_DIR}")

# [Copy CELL 3-10 from the original file - they're the same]
# ... (all the function definitions)

# ============================================================================
# CELL 10: Filtering (MODIFIED - More lenient)
# ============================================================================

def filter_garbage(dataset: Dataset) -> Dataset:
    """Filter out garbage/invalid rows - MORE LENIENT"""
    def is_valid(example):
        # Must have minimum text length
        if not example['text'] or len(example['text']) < MIN_TEXT_LENGTH:
            return False
        
        # LENIENT: Accept rows even if timestamp parsing failed
        # This allows logs where timestamp is embedded in content but couldn't be parsed
        
        # Check if mostly binary/noise
        text = example['text']
        alnum_count = sum(c.isalnum() for c in text[:200])
        if alnum_count < len(text[:200]) * 0.3:
            return False
        
        return True
    
    return dataset.filter(is_valid, desc="Filtering garbage")

