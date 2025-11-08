# Quick Start - Log Cleaning Pipeline

## 🎯 Goal
Clean 5.4M logs from `chYassine/ait-fox-raw-v02` for log parsing, summarization, and classification.

## ⚡ Run in Google Colab (3 Commands)

### 1. Install and Setup
```python
!pip -q install datasets pandas tqdm
```

### 2. Copy-Paste Pipeline
Open `notebooks/clean_logs_colab.py` and copy all cells into Colab, then:
```python
# Runtime → Run all
```

### 3. Download Results
```python
from google.colab import files
import shutil

shutil.make_archive('cleaned', 'zip', 'datasets/ait/output/ait/cleaned')
files.download('cleaned.zip')
```

## 📊 What You'll Get

- **~4.4M cleaned logs** (from 5.4M raw)
- **27 columns** including:
  - `text` - normalized, redacted log text
  - `text_canonical` - template-ready with placeholders
  - `timestamp` - UTC ISO8601
  - `log_type_canonical` - standardized log type
  - `rounded_ts_30m` - for session grouping
  - `structured_fields` - extracted JSON/key-value
  - `severity` - extracted severity level

## ⏱️ Timeline

| Step | Time | Output |
|------|------|--------|
| Install deps | 1 min | - |
| Load dataset | 2 min | 5.4M rows loaded |
| Process | 15 min | All fields normalized |
| Filter & dedup | 5 min | ~1M duplicates removed |
| Save | 3 min | Dataset + artifacts |
| **Total** | **~30 min** | **Ready for Phase 1** |

## ✅ Quality Checks

After running, verify:

```python
from datasets import load_from_disk
import json

ds = load_from_disk("datasets/ait/output/ait/cleaned")
with open("artifacts/cleaning/stats.json") as f:
    stats = json.load(f)

print(f"Rows: {len(ds):,}")
print(f"Timestamp parse: {stats['timestamp_parse_rate']:.1%}")  # Should be >99%
print(f"Avg text len: {stats['avg_text_length']:.0f}")
print(f"Top log types: {list(stats['log_type_counts'].keys())[:5]}")
```

Expected output:
```
Rows: 4,400,000-4,500,000
Timestamp parse: 99.5%+
Avg text len: 250-350
Top log types: ['suricata', 'logstash', 'apache2', 'journal', 'horde']
```

## 🎨 What Gets Cleaned

| Aspect | Before | After |
|--------|--------|-------|
| Timestamps | Mixed formats | UTC ISO8601 |
| Text | Control chars, mixed encoding | NFC normalized |
| IPs/Emails | `192.168.1.1`, `user@domain` | `<IP:abc12345>`, `<EMAIL:def67890>` |
| Duplicates | ~20% redundant | Deduplicated |
| Templates | Raw text | `<IP> <NUM> [<TS>] "<PATH>"` |
| Sessions | Manual grouping | `rounded_ts_30m` ready |

## 📁 Key Files

### Created by Pipeline
- `datasets/ait/output/ait/cleaned/` - Main dataset (HF format)
- `artifacts/cleaning/stats.json` - Statistics
- `artifacts/cleaning/quality_report.md` - Human-readable report
- `artifacts/cleaning/samples/sample_train.csv` - 100-row sample

### Guides (Already Created)
- `COLAB_CLEANING_GUIDE.md` - Detailed Colab guide
- `DATA_CLEANING_PLAN_SUMMARY.md` - Full implementation summary
- `datasets/ait/output/ait/cleaned/README.md` - Dataset documentation
- `PHASE1_TRAINING_README.md` - Next steps

### Code
- `notebooks/clean_logs_colab.py` - **Use this for Colab**
- `scripts/clean_hf_logs.py` - Standalone script (if running locally)

## 🔗 Next Steps

1. ✅ Run cleaning pipeline (you are here)
2. ⏭️ Run Phase 1 preprocessing: `python scripts/preprocess_phase1_training.py`
3. ⏭️ Train base model on preprocessed data

## 🆘 Common Issues

**Issue:** `ImportError: cannot import name 'BeamWriter'`  
**Fix:** Restart runtime, then run:
```python
!pip -q install -U pyarrow datasets==2.19.1
```

**Issue:** Out of memory  
**Fix:** Process subset first:
```python
dataset = load_dataset(HF_REPO, split="train[:10%]")
```

**Issue:** Low timestamp parse rate (<95%)  
**Fix:** Check `artifacts/cleaning/quality_report.md` for patterns, add custom formats

## 📚 Learn More

- **Detailed guide:** `COLAB_CLEANING_GUIDE.md`
- **Implementation details:** `DATA_CLEANING_PLAN_SUMMARY.md`
- **Dataset docs:** `datasets/ait/output/ait/cleaned/README.md`

---

**Ready to run?** Open `notebooks/clean_logs_colab.py` in Colab and execute all cells!

**Questions?** Check `COLAB_CLEANING_GUIDE.md` for troubleshooting.

