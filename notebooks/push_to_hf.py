# PUSH CLEANED DATASET TO HUGGINGFACE
# Run this in Colab after the cleaning pipeline completes

from datasets import load_from_disk
from huggingface_hub import login

# ============================================================================
# STEP 1: Login to HuggingFace
# ============================================================================

print("Please login to HuggingFace:")
login()  # This will prompt for your token

# ============================================================================
# STEP 2: Load cleaned dataset
# ============================================================================

OUT_DIR = "datasets/ait/output/ait/cleaned"
HF_OUTPUT_REPO = "chYassine/ait-cleaned-logs-v1"  # Change this to your desired repo name

print(f"\nLoading cleaned dataset from {OUT_DIR}...")
dataset = load_from_disk(OUT_DIR)
print(f"✓ Loaded {len(dataset):,} rows")

# ============================================================================
# STEP 3: Push to HuggingFace
# ============================================================================

print(f"\nPushing to HuggingFace: {HF_OUTPUT_REPO}...")
print("This may take 5-10 minutes...")

dataset.push_to_hub(
    HF_OUTPUT_REPO,
    private=False,  # Set to True if you want a private dataset
    commit_message="Initial upload of cleaned AIT logs"
)

print(f"\n✅ Dataset successfully pushed to HuggingFace!")
print(f"   View at: https://huggingface.co/datasets/{HF_OUTPUT_REPO}")

# ============================================================================
# STEP 4: Verify upload
# ============================================================================

from datasets import load_dataset

print("\nVerifying upload...")
ds_verify = load_dataset(HF_OUTPUT_REPO, split="train")
print(f"✓ Verified: {len(ds_verify):,} rows in HuggingFace dataset")
print(f"✓ Columns: {ds_verify.column_names}")

