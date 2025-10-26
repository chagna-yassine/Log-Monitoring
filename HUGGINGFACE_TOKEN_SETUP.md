# Hugging Face Token Setup Guide

This guide will help you set up your Hugging Face token so you can upload processed AIT datasets.

## 🔑 Why Do You Need a Token?

Hugging Face requires authentication to upload datasets. The token proves you have permission to upload to your repositories.

## 📋 Quick Setup Options

### Option 1: Automated Setup Scripts

**Windows:**
```bash
scripts/setup_hf_token.bat
```

**Linux/Mac:**
```bash
scripts/setup_hf_token.sh
```

**Python (Cross-platform):**
```bash
python scripts/setup_hf_token.py
```

### Option 2: Manual Setup

#### Step 1: Get Your Token

1. Go to [Hugging Face Settings](https://huggingface.co/settings/tokens)
2. Click **"New token"**
3. Give it a name (e.g., `ait-processing`)
4. Select **"Write"** permissions
5. Click **"Generate token"**
6. **Copy the token** (you won't see it again!)

#### Step 2: Set Environment Variable

**Windows (Command Prompt):**
```cmd
set HUGGINGFACE_HUB_TOKEN=your_token_here
```

**Windows (PowerShell):**
```powershell
$env:HUGGINGFACE_HUB_TOKEN="your_token_here"
```

**Linux/Mac:**
```bash
export HUGGINGFACE_HUB_TOKEN="your_token_here"
```

#### Step 3: Make It Permanent

**Windows:**
- Add to System Environment Variables
- Or add to your PowerShell profile

**Linux/Mac:**
```bash
echo 'export HUGGINGFACE_HUB_TOKEN="your_token_here"' >> ~/.bashrc
source ~/.bashrc
```

### Option 3: Hugging Face CLI

```bash
# Install CLI
pip install huggingface_hub

# Login
huggingface-cli login

# Enter your token when prompted
```

## 🧪 Test Your Setup

Run this Python code to test:

```python
import os
from huggingface_hub import HfApi

# Check if token is set
token = os.getenv('HUGGINGFACE_HUB_TOKEN')
if token:
    print(f"✅ Token found: {token[:10]}...{token[-4:]}")
    
    # Test authentication
    api = HfApi()
    user_info = api.whoami()
    print(f"✅ Authenticated as: {user_info['name']}")
else:
    print("❌ No token found")
```

## 🚀 Ready to Upload!

Once your token is set up, you can run the chunked processing:

```bash
python scripts/preprocess_ait_chunked.py
```

The script will:
1. ✅ Check for your token
2. ✅ Verify authentication
3. ✅ Upload each chunk to Hugging Face
4. ✅ Delete local files to save space

## 🔧 Troubleshooting

### "401 Unauthorized" Error
- **Cause**: No token or invalid token
- **Solution**: Set up your token using one of the methods above

### "Repository Not Found" Error
- **Cause**: Repository doesn't exist yet
- **Solution**: The script will create it automatically with your token

### "Invalid username or password" Error
- **Cause**: Token is incorrect or expired
- **Solution**: Generate a new token and update your environment variable

### Token Not Persisting
- **Cause**: Environment variable only set for current session
- **Solution**: Add to your shell profile or system environment variables

## 📚 Additional Resources

- [Hugging Face Token Documentation](https://huggingface.co/docs/hub/security-tokens)
- [Hugging Face CLI Documentation](https://huggingface.co/docs/huggingface_hub/quick-start)
- [Environment Variables Guide](https://docs.python.org/3/library/os.html#os.environ)

## 🆘 Need Help?

If you're still having issues:

1. **Check your token**: Make sure it's copied correctly
2. **Verify permissions**: Token must have "Write" permissions
3. **Test manually**: Use the test code above
4. **Check environment**: Make sure the variable is set correctly

The chunked processing will work much better once authentication is set up! 🎉
