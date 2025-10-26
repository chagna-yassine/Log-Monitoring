@echo off
echo ========================================
echo HUGGING FACE TOKEN SETUP (Windows)
echo ========================================
echo.
echo This script will help you set up your Hugging Face token.
echo.
echo 1. Go to: https://huggingface.co/settings/tokens
echo 2. Click 'New token'
echo 3. Give it a name (e.g., 'ait-processing')
echo 4. Select 'Write' permissions
echo 5. Copy the token
echo.
set /p HF_TOKEN="Enter your Hugging Face token: "

if "%HF_TOKEN%"=="" (
    echo Error: No token provided.
    pause
    exit /b 1
)

echo.
echo Setting token for current session...
set HUGGINGFACE_HUB_TOKEN=%HF_TOKEN%

echo.
echo Testing token...
python -c "import os; from huggingface_hub import HfApi; api = HfApi(); print('✅ Token works! Authenticated as:', api.whoami()['name'])"

if %errorlevel% equ 0 (
    echo.
    echo ✅ SUCCESS! Your token is working.
    echo.
    echo To make this permanent, add this line to your environment variables:
    echo HUGGINGFACE_HUB_TOKEN=%HF_TOKEN%
    echo.
    echo Or run this command in PowerShell:
    echo [Environment]::SetEnvironmentVariable("HUGGINGFACE_HUB_TOKEN", "%HF_TOKEN%", "User")
) else (
    echo.
    echo ❌ Token test failed. Please check your token and try again.
)

echo.
pause
