#!/bin/bash
# Download Harrison dataset from Zenodo

echo "======================================================================"
echo "DOWNLOADING HARRISON AIT DATASET"
echo "======================================================================"

# Configuration
DATASET_NAME="harrison"
BASE_URL="https://zenodo.org/records/5789064/files"
DATASET_DIR="datasets/ait"

echo ""
echo "Dataset: Harrison"
echo "URL: ${BASE_URL}/${DATASET_NAME}.zip"
echo "Target: ${DATASET_DIR}"
echo ""

# Create directory if it doesn't exist
mkdir -p "$DATASET_DIR"

# Download the dataset
echo "Downloading Harrison dataset..."
echo "URL: ${BASE_URL}/${DATASET_NAME}.zip?download=1"
echo ""

curl -L -o "${DATASET_DIR}/${DATASET_NAME}.zip" "${BASE_URL}/${DATASET_NAME}.zip?download=1"

# Check if download was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Download complete!"
    echo "File saved: ${DATASET_DIR}/${DATASET_NAME}.zip"
    
    # Extract the zip file
    echo ""
    echo "Extracting dataset..."
    unzip -q "${DATASET_DIR}/${DATASET_NAME}.zip" -d "${DATASET_DIR}"
    
    # Remove zip file after extraction
    if [ $? -eq 0 ]; then
        echo "✅ Extraction complete!"
        rm "${DATASET_DIR}/${DATASET_NAME}.zip"
        echo "✅ Cleaned up zip file"
    else
        echo "⚠️  Extraction failed"
    fi
else
    echo ""
    echo "❌ Download failed!"
    exit 1
fi

echo ""
echo "======================================================================"
echo "HARRISON DATASET READY!"
echo "======================================================================"
echo "Location: ${DATASET_DIR}/${DATASET_NAME}"
echo ""
echo "Now run:"
echo "  python scripts/upload_harrison_logs.py"
echo "======================================================================"

