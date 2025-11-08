# FILTER GARBAGE AND COUNT TOKENS
# Run this in Colab after initial cleaning

from datasets import load_dataset, load_from_disk, Dataset
from huggingface_hub import login
import re
from tqdm import tqdm

# ============================================================================
# STEP 1: Load cleaned dataset
# ============================================================================

print("Loading cleaned dataset from HuggingFace...")
dataset = load_dataset("chYassine/ait-cleaned-logs-v1", split="train")
print(f"Initial rows: {len(dataset):,}")

# ============================================================================
# STEP 2: Define garbage filters
# ============================================================================

def is_garbage(text: str) -> bool:
    """Detect garbage/corrupted logs using multiple heuristics"""
    if not text or len(text) < 3:
        return True
    
    # Heuristic 1: Too many repeated characters
    if len(text) >= 3:
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
        max_repeat = max(char_counts.values())
        # If a single character appears more than 70% of the time
        if max_repeat / len(text) > 0.7:
            return True
    
    # Heuristic 2: Check for common garbage patterns
    garbage_patterns = [
        r'^[A-Za-z0-9\[\]\{\}\(\)\<\>\?\!@#\$%\^\&\*\+\=\|\\\/~`;\:,\."\s]{1,10}$',  # Short random strings
        r'^[^\x20-\x7E]*$',  # Only non-printable ASCII
        r'^[\x00-\x1F]{2,}$',  # Only control characters
    ]
    
    for pattern in garbage_patterns:
        if re.match(pattern, text):
            # Additional check: if it's very short and matches garbage pattern, it's garbage
            if len(text) <= 10:
                return True
    
    # Heuristic 3: Check entropy (random garbage has high entropy)
    # Calculate character entropy
    if len(text) > 0:
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        import math
        entropy = 0
        for count in char_counts.values():
            p = count / len(text)
            if p > 0:
                entropy -= p * math.log2(p)  # Shannon entropy
        
        # Very high entropy with very low length = likely garbage
        if len(text) <= 20 and entropy > math.log2(len(char_counts)) * 0.8:
            return True
    
    # Heuristic 4: Check if text has ANY structure (legitimate logs usually do)
    # Look for at least one word-like pattern (3+ consecutive alnum chars)
    if not re.search(r'[A-Za-z0-9]{3,}', text):
        # No word-like patterns found = likely garbage
        return True
    
    # Heuristic 5: Check for common log structures
    # Legitimate logs usually have at least one of:
    # - A timestamp pattern
    # - A common word (GET, POST, ERROR, INFO, etc.)
    # - An IP address
    # - A URL/path
    # - A number > 10
    
    has_structure = False
    
    # Check for timestamp
    if re.search(r'\d{4}-\d{2}-\d{2}|\d{2}/\w{3}/\d{4}|[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}', text):
        has_structure = True
    
    # Check for common log words
    common_words = ['GET', 'POST', 'PUT', 'DELETE', 'ERROR', 'WARN', 'INFO', 'DEBUG', 'TRACE',
                    'access', 'denied', 'allowed', 'success', 'failed', 'timeout', 'connection',
                    'request', 'response', 'server', 'client', 'host', 'port', 'src', 'dst']
    if any(word in text.upper() for word in common_words):
        has_structure = True
    
    # Check for IP address
    if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', text):
        has_structure = True
    
    # Check for URL/path
    if re.search(r'https?://|/[a-zA-Z0-9_\-]+/', text):
        has_structure = True
    
    # Check for meaningful number
    if re.search(r'\b\d{2,}\b', text):  # Number with 2+ digits
        has_structure = True
    
    # If no structure found and text is short, likely garbage
    if not has_structure and len(text) < 50:
        return True
    
    return False


# ============================================================================
# STEP 3: Filter garbage
# ============================================================================

print("\nFiltering garbage logs...")
initial_count = len(dataset)

def filter_garbage_rows(example):
    text = example.get('text', '') or example.get('text_original', '')
    # Also check original text for better detection
    return not is_garbage(text)

dataset_filtered = dataset.filter(
    filter_garbage_rows,
    desc="Removing garbage logs"
)

filtered_count = initial_count - len(dataset_filtered)
print(f"✓ Removed {filtered_count:,} garbage rows ({filtered_count/initial_count:.1%})")
print(f"✓ Remaining rows: {len(dataset_filtered):,}")

# ============================================================================
# STEP 4: Count tokens
# ============================================================================

print("\nCounting tokens...")
print("Using Qwen2.5-72B tokenizer")

try:
    from transformers import AutoTokenizer
    import tiktoken
    
    # Load Qwen tokenizer
    print("Loading Qwen tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-72B-Instruct", trust_remote_code=True)
    
    total_tokens = 0
    total_chars = 0
    
    print("Counting tokens with Qwen tokenizer...")
    for row in tqdm(dataset_filtered, desc="Counting"):
        text = row.get('text', '')
        if text:
            # Use Qwen tokenizer
            tokens = tokenizer.encode(text, add_special_tokens=False)
            total_tokens += len(tokens)
            total_chars += len(text)
    
    print(f"\n{'='*60}")
    print(f"TOKEN COUNT SUMMARY")
    print(f"{'='*60}")
    print(f"Tokenizer: Qwen2.5-72B-Instruct")
    print(f"Total rows: {len(dataset_filtered):,}")
    print(f"Total characters: {total_chars:,}")
    print(f"Total tokens (Qwen): {total_tokens:,}")
    print(f"Average tokens per log: {total_tokens/len(dataset_filtered):.1f}")
    print(f"Average chars per log: {total_chars/len(dataset_filtered):.1f}")
    print(f"Chars/token ratio: {total_chars/total_tokens:.2f}")
    
    # Cost estimation (rough)
    print(f"\n{'='*60}")
    print(f"COST ESTIMATION (rough)")
    print(f"{'='*60}")
    print(f"For Qwen 72B fine-tuning:")
    print(f"Estimated cost: ${total_tokens / 1_000_000 * 8:.2f} (assumes ~$8 per 1M tokens)")
    
    print(f"\nComparison with GPT-4 tokenizer:")
    gpt4_tokens = len(tiktoken.get_encoding("cl100k_base").encode(" ".join([row.get('text', '') for row in dataset_filtered.select(range(10000))])))
    avg_gpt4 = gpt4_tokens / 10000
    print(f"Sample avg (GPT-4 tokenizer): ~{avg_gpt4:.1f} tokens/log")
    print(f"Your avg (Qwen tokenizer): {total_tokens/len(dataset_filtered):.1f} tokens/log")

except ImportError:
    print("transformers not installed, installing...")
    import subprocess
    subprocess.check_call(["pip", "install", "transformers", "tiktoken"])
    
    from transformers import AutoTokenizer
    
    print("Loading Qwen tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-72B-Instruct", trust_remote_code=True)
    
    total_tokens = 0
    total_chars = 0
    
    for row in tqdm(dataset_filtered, desc="Counting"):
        text = row.get('text', '')
        if text:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            total_tokens += len(tokens)
            total_chars += len(text)
    
    print(f"\n{'='*60}")
    print(f"TOKEN COUNT SUMMARY (Qwen2.5-72B)")
    print(f"{'='*60}")
    print(f"Total rows: {len(dataset_filtered):,}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Average tokens per log: {total_tokens/len(dataset_filtered):.1f}")

# ============================================================================
# STEP 5: Show examples of filtered garbage
# ============================================================================

print(f"\n{'='*60}")
print(f"SAMPLE OF FILTERED GARBAGE")
print(f"{'='*60}")

# Re-load original to find what was filtered
dataset_orig = load_dataset("chYassine/ait-cleaned-logs-v1", split="train")

garbage_samples = []
for row in dataset_orig:
    text = row.get('text', '')
    if is_garbage(text):
        garbage_samples.append({
            'text': text,
            'host': row.get('host_sanitized'),
            'log_type': row.get('log_type_canonical')
        })
        if len(garbage_samples) >= 20:
            break

for i, sample in enumerate(garbage_samples[:10], 1):
    print(f"\n{i}. Host: {sample['host']}, Type: {sample['log_type']}")
    print(f"   Text: {repr(sample['text'])}")

# ============================================================================
# STEP 6: Save filtered dataset
# ============================================================================

print(f"\n{'='*60}")
print(f"SAVING FILTERED DATASET")
print(f"{'='*60}")

OUT_DIR_FILTERED = "datasets/ait/output/ait/cleaned_filtered"
dataset_filtered.save_to_disk(OUT_DIR_FILTERED)
print(f"✓ Saved to {OUT_DIR_FILTERED}")

# ============================================================================
# STEP 7: Push to HuggingFace
# ============================================================================

print(f"\n{'='*60}")
print(f"PUSHING TO HUGGINGFACE")
print(f"{'='*60}")

try:
    login()  # Will prompt for token if not already logged in
    
    HF_REPO = "chYassine/ait-cleaned-logs-v1"
    print(f"Pushing to {HF_REPO}...")
    
    dataset_filtered.push_to_hub(
        HF_REPO,
        private=False,
        commit_message=f"Cleaned and filtered: {len(dataset_filtered):,} rows, {total_tokens:,} tokens"
    )
    
    print(f"✅ Successfully pushed to HuggingFace!")
    print(f"   URL: https://huggingface.co/datasets/{HF_REPO}")
    
except Exception as e:
    print(f"⚠️ Error pushing to HuggingFace: {e}")
    print(f"   Dataset saved locally at: {OUT_DIR_FILTERED}")

# ======================================================================0======
# FINAL SUMMARY
# ============================================================================

print(f"\n{'='*60}")
print(f"FINAL SUMMARY")
print(f"{'='*60}")
print(f"✓ Removed {filtered_count:,} garbage rows")
print(f"✓ {len(dataset_filtered):,} clean rows remaining")
print(f"✓ Total tokens: {total_tokens:,}")
print(f"✓ Dataset saved and pushed to HuggingFace")
print(f"{'='*60}")

