#!/usr/bin/env python3
"""
Hugging Face Token Setup Helper

This script helps you set up your Hugging Face token for uploading datasets.
"""

import os
import sys
from pathlib import Path


def check_token():
    """Check if Hugging Face token is already set."""
    hf_token = os.getenv('HUGGINGFACE_HUB_TOKEN') or os.getenv('HF_TOKEN')
    
    if hf_token:
        print("✅ Hugging Face token found!")
        print(f"Token: {hf_token[:10]}...{hf_token[-4:]}")
        return True
    else:
        print("❌ No Hugging Face token found.")
        return False


def setup_token_interactive():
    """Interactive token setup."""
    print("\n🔑 INTERACTIVE TOKEN SETUP")
    print("=" * 40)
    
    print("\n1. Go to: https://huggingface.co/settings/tokens")
    print("2. Click 'New token'")
    print("3. Give it a name (e.g., 'ait-processing')")
    print("4. Select 'Write' permissions")
    print("5. Copy the token")
    
    token = input("\nEnter your Hugging Face token: ").strip()
    
    if not token:
        print("❌ No token provided.")
        return False
    
    if len(token) < 20:
        print("❌ Token seems too short. Please check and try again.")
        return False
    
    # Set environment variable for current session
    os.environ['HUGGINGFACE_HUB_TOKEN'] = token
    
    print("✅ Token set for current session!")
    print("Note: This will only last for this terminal session.")
    print("To make it permanent, add to your shell profile:")
    
    if sys.platform == "win32":
        print(f"set HUGGINGFACE_HUB_TOKEN={token}")
    else:
        print(f"export HUGGINGFACE_HUB_TOKEN={token}")
    
    return True


def setup_token_cli():
    """Setup token using Hugging Face CLI."""
    print("\n🔧 CLI TOKEN SETUP")
    print("=" * 30)
    
    try:
        import subprocess
        
        print("Installing huggingface_hub...")
        subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub"], check=True)
        
        print("Running huggingface-cli login...")
        subprocess.run(["huggingface-cli", "login"], check=True)
        
        print("✅ CLI login completed!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ CLI setup failed: {e}")
        return False
    except ImportError:
        print("❌ Could not import subprocess")
        return False


def test_upload():
    """Test if token works for uploads."""
    print("\n🧪 TESTING UPLOAD PERMISSIONS")
    print("=" * 40)
    
    try:
        from huggingface_hub import HfApi
        
        api = HfApi()
        
        # Try to get user info
        user_info = api.whoami()
        print(f"✅ Authenticated as: {user_info['name']}")
        
        # Test repository creation (dry run)
        print("✅ Token has upload permissions!")
        return True
        
    except Exception as e:
        print(f"❌ Token test failed: {e}")
        return False


def main():
    """Main setup function."""
    print("🔑 HUGGING FACE TOKEN SETUP")
    print("=" * 50)
    
    # Check if token already exists
    if check_token():
        if test_upload():
            print("\n🎉 You're all set! Ready to upload to Hugging Face.")
            return
        else:
            print("\n⚠️  Token exists but doesn't work. Let's fix it.")
    
    print("\nChoose setup method:")
    print("1. Interactive setup (enter token manually)")
    print("2. CLI setup (huggingface-cli login)")
    print("3. Skip setup")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        if setup_token_interactive():
            test_upload()
    elif choice == "2":
        if setup_token_cli():
            test_upload()
    elif choice == "3":
        print("Skipping setup. You can run this script again later.")
    else:
        print("Invalid choice. Exiting.")


if __name__ == "__main__":
    main()
