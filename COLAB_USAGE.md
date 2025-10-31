# Using the Analysis Script in Google Colab

## ✅ Updated for Single Dataset Analysis

To avoid memory issues, the script now analyzes ONE dataset at a time. Works perfectly in Colab!

## Quick Start in Colab

### 1. Upload the Script

First, upload `scripts/analyze_all_ait_datasets.py` to your Colab environment:

```python
# If you haven't already, install dependencies
!pip install datasets huggingface_hub pandas -q
```

### 2. Run the Script

**Method 1** - Direct function call:

```python
# Copy the script or import it
from analyze_all_ait_datasets import analyze_dataset

# Analyze ONE dataset
analyze_dataset('chYassine/ait-fox-raw-v02')
```

**Method 2** - Upload the file and execute:

```python
# Upload the script file
from google.colab import files
uploaded = files.upload()

# Import and run
import analyze_all_ait_datasets

# Analyze a specific dataset
analyze_all_ait_datasets.analyze_dataset('chYassine/ait-fox-raw-v02')
```

**Method 3** - With authentication:

```python
# Install and authenticate
!pip install datasets huggingface_hub pandas -q

from huggingface_hub import login
login(token="YOUR_TOKEN_HERE")

# Then analyze
from analyze_all_ait_datasets import analyze_dataset
analyze_dataset('chYassine/ait-fox-raw-v02')
```

## Complete Example Cell

```python
# Install dependencies
!pip install datasets huggingface_hub pandas -q

# Authenticate with HuggingFace (if datasets are private)
from huggingface_hub import login
login(token="YOUR_TOKEN_HERE")

# Import the analysis function
import sys
sys.path.append('.')
from analyze_all_ait_datasets import analyze_dataset

# Run analysis on ONE dataset
analyze_dataset('chYassine/ait-fox-raw-v02')
```

## Analyze Multiple Datasets (One at a Time)

If you want to analyze all datasets, run them sequentially:

```python
datasets = [
    'chYassine/ait-wilson-raw-v01',
    'chYassine/ait-wheeler-raw-v01',
    'chYassine/ait-wardbeck-raw-v01',
    'chYassine/ait-shaw-raw-v01',
    'chYassine/ait-santos-raw-v01',
    'chYassine/air-russellmitchell-raw-v01',
    'chYassine/ait-harrison-raw-v01',
    'chYassine/ait-fox-raw-v02'
]

for dataset in datasets:
    print(f"\n{'='*80}")
    print(f"Analyzing: {dataset}")
    print('='*80)
    analyze_dataset(dataset)
    # Outputs will be saved separately for each dataset
```

## Alternative: Clone from Git

```python
# Clone the repository
!git clone https://github.com/yourusername/yourrepo.git
%cd yourrepo

# Run the script
!python scripts/analyze_all_ait_datasets.py
```

## What Changed?

The script has been updated for better performance:

1. **Single dataset analysis** - Analyzes one dataset at a time to avoid memory issues
2. **Token input** - Interactive prompt for HuggingFace authentication
3. **Jupyter/Colab compatible** - Handles kernel arguments gracefully
4. **Memory efficient** - No longer loads all 8 datasets simultaneously

## Authentication

If your datasets are private:

```python
from huggingface_hub import login

# Option 1: Use token from environment
import os
login(token=os.environ['HUGGINGFACE_HUB_TOKEN'])

# Option 2: Enter token directly
login(token="hf_xxxxxxxxxxxx")
```

Or set as secret in Colab:
1. Add secrets in Colab (🔑 icon on left sidebar)
2. Add `HUGGINGFACE_HUB_TOKEN` with your token
3. Access: `from google.colab import userdata; login(token=userdata.get('HUGGINGFACE_HUB_TOKEN'))`

## Output

The script will:
- ✅ Load all 8 datasets
- ✅ Print comprehensive statistics
- ✅ Show sample logs
- ✅ Save `ait_datasets_analysis_summary.json`

## Troubleshooting

### Import Errors

```python
!pip install datasets huggingface_hub pandas
```

### Authentication Errors

Make sure you're logged in:
```python
from huggingface_hub import login
login()
```

### Memory Issues

The script now only analyzes one dataset at a time, which should prevent memory issues. If you still encounter problems:

1. Restart runtime and clear output
2. Use a smaller dataset first
3. Enable High-RAM runtime (Runtime → Change runtime type → High-RAM)

### Slow First Run

First time downloads datasets (~2-10 min depending on size). Subsequent runs use cache.

## Tips for Colab

1. **Use GPU**: Enable GPU runtime for faster processing (Settings → Hardware Accelerator → GPU)
2. **Save Results**: Download the JSON summary before session ends
3. **Monitor Memory**: Check RAM usage in Colab's resource monitor
4. **Cache**: Datasets are cached, so second run is much faster

## Next Steps

After analysis:
1. Review the output in Colab
2. Download `ait_datasets_analysis_summary.json`
3. Use insights for preprocessing/training
4. Share findings with your team

---

**Ready to Analyze!** 🚀

