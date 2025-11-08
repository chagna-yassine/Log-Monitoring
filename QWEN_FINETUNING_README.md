# Qwen2.5-72B Fine-tuning for Log Understanding

This directory contains a comprehensive Jupyter notebook for fine-tuning **Qwen/Qwen2.5-72B-Instruct** with quantization and Soft Mixture of Experts (SoftMoE) for log understanding tasks.

## 📁 Files

- **`finetune_qwen_softmoe_logs.ipynb`**: Main Colab notebook for fine-tuning
- **`generate_finetune_notebook.py`**: Script to regenerate the notebook if needed

## 🎯 What This Notebook Does

The notebook implements a complete training pipeline for fine-tuning Qwen2.5-72B-Instruct on three log understanding tasks:

1. **Log Parsing**: Extract templates from raw log lines by replacing variable parts
2. **Log Classification**: Classify logs as normal or anomaly
3. **Log Summarization**: Generate summaries of log sequences

## 🚀 Quick Start

### 1. Upload to Google Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click `File` → `Upload notebook`
3. Upload `finetune_qwen_softmoe_logs.ipynb`
4. Set runtime to GPU: `Runtime` → `Change runtime type` → `GPU` (T4, V100, or A100)

### 2. Setup HuggingFace Token

Add your HuggingFace token to Colab secrets:
1. Click the 🔑 (key) icon in the left sidebar
2. Add a new secret: name = `HF_TOKEN`, value = your token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Enable notebook access

### 3. Configure Dataset

Edit the configuration cell to point to your HuggingFace dataset:

```python
@dataclass
class Config:
    # ... other settings ...
    
    # Dataset settings - CHANGE THIS TO YOUR HUGGINGFACE DATASET
    dataset_name: str = "chYassine/ait-fox-raw-logs"  # Your HF dataset
```

### 4. Run All Cells

Click `Runtime` → `Run all` and let the notebook train!

## 📊 Key Features

### 1. Quantization (QLoRA)

- **4-bit quantization** using bitsandbytes
- **NF4 (Normal Float 4)** quantization type
- **Double quantization** for extra memory savings
- Reduces 72B model from ~144GB to ~18GB of VRAM

### 2. Soft Mixture of Experts (SoftMoE)

- **8 expert networks** specialized for different log patterns
- **Top-2 routing** per token
- **Soft routing** with smooth gradient flow
- Better than hard routing for multi-task learning

### 3. Multi-Task Learning

The notebook automatically generates training data for three tasks:
- **40%** Log Parsing tasks
- **40%** Log Classification tasks  
- **20%** Log Summarization tasks

### 4. LoRA Adapters

- **Rank 64** with alpha 128
- Targets all attention and MLP layers
- Only ~0.1% of parameters are trainable
- Fast training with minimal memory

## ⚙️ Configuration Options

### Memory Settings

If you run out of memory, adjust these:

```python
class Config:
    max_seq_length: int = 2048  # Reduce to 1024 or 512
    per_device_train_batch_size: int = 1  # Keep at 1
    gradient_accumulation_steps: int = 16  # Increase for larger effective batch
```

### Training Settings

```python
class Config:
    num_train_epochs: int = 3  # Number of training epochs
    learning_rate: float = 2e-5  # Learning rate
    warmup_steps: int = 100  # Warmup steps
```

### LoRA Settings

```python
class Config:
    lora_r: int = 64  # LoRA rank (higher = more capacity)
    lora_alpha: int = 128  # LoRA scaling factor
    lora_dropout: float = 0.1  # Dropout rate
```

### SoftMoE Settings

```python
class Config:
    num_experts: int = 8  # Number of expert networks
    num_experts_per_token: int = 2  # Active experts per token
    use_softmoe: bool = True  # Enable/disable SoftMoE
```

## 📝 Dataset Requirements

Your HuggingFace dataset should have at least one of these columns:

- **`content`** (required): The raw log text
- **`host`** (optional): Host/server name
- **`log_type`** (optional): Type of log (e.g., apache, syslog)

Example dataset structure:

```python
{
    "content": "2024-01-15 10:00:00 INFO Service starting on port 8080",
    "host": "webserver-01",
    "log_type": "application"
}
```

## 💡 Training Tips

### GPU Recommendations

- **T4 (Free Colab)**: Works but slower (~2-3 hours for 2000 examples)
- **V100 (Colab Pro)**: Good performance (~1 hour)
- **A100 (Colab Pro+)**: Fastest (~30 minutes)

### Memory Optimization

1. **Reduce sequence length**: `max_seq_length = 1024` instead of 2048
2. **Increase gradient accumulation**: `gradient_accumulation_steps = 32`
3. **Enable gradient checkpointing** (uncomment in notebook)

### Training Duration

Expected training time for 2000 examples (3 epochs):
- T4: 2-3 hours
- V100: 1 hour
- A100: 30 minutes

## 📤 Saving Your Model

The notebook saves the model to `./qwen-log-understanding/` by default.

### Save to Google Drive

Add this to the config:

```python
class Config:
    output_dir: str = "/content/drive/MyDrive/qwen-log-understanding"
```

### Upload to HuggingFace Hub

Set `upload_to_hub = True` in the last cell and provide your repo name.

## 🧪 Testing the Model

The notebook includes a testing section that runs inference on sample logs for all three tasks:

```python
# Test all task types
for task in ['parsing', 'classification', 'summarization']:
    test_model(task)
```

This will show you how well the model performs on:
- Extracting log templates
- Classifying anomalies
- Summarizing log sequences

## 🔧 Troubleshooting

### Out of Memory

**Error**: `CUDA out of memory`

**Solution**: Reduce `max_seq_length` to 1024 or 512:

```python
config.max_seq_length = 1024
```

### Slow Training

**Issue**: Training is taking too long

**Solution**: 
1. Reduce `num_train_epochs` to 1 or 2
2. Use fewer examples: `num_examples=1000`
3. Upgrade to V100 or A100 GPU

### Model Not Loading

**Error**: `HuggingFaceTokenError` or authentication issues

**Solution**:
1. Check your HF_TOKEN is correct
2. Accept Qwen2.5 license at [huggingface.co/Qwen/Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct)

### Dataset Not Found

**Error**: `DatasetNotFoundError`

**Solution**:
1. Verify dataset name is correct
2. Ensure dataset is public or you have access
3. Use the sample dataset (notebook will create automatically)

## 📚 Technical Details

### Architecture

- **Base Model**: Qwen2.5-72B-Instruct (72 billion parameters)
- **Quantization**: 4-bit NF4 with double quantization
- **Training Method**: QLoRA (Quantized Low-Rank Adaptation)
- **Expert System**: Soft Mixture of Experts (8 experts, top-2 routing)

### Training Data Format

The notebook uses Qwen's chat format:

```
<|im_start|>system
You are Qwen, an AI assistant specialized in log analysis and understanding.<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
{response}<|im_end|>
```

### Multi-Task Distribution

Training examples are automatically generated with:
- 40% parsing tasks
- 40% classification tasks
- 20% summarization tasks

## 📖 References

### Papers

- **Qwen2.5**: [GitHub - QwenLM/Qwen](https://github.com/QwenLM/Qwen)
- **QLoRA**: [arxiv.org/abs/2305.14314](https://arxiv.org/abs/2305.14314)
- **SoftMoE**: [arxiv.org/abs/2308.00951](https://arxiv.org/abs/2308.00951)

### Libraries

- **Transformers**: Hugging Face library for LLMs
- **PEFT**: Parameter-Efficient Fine-Tuning
- **bitsandbytes**: 4-bit/8-bit quantization
- **accelerate**: Distributed training

## 🤝 Contributing

To regenerate the notebook with modifications:

```bash
python generate_finetune_notebook.py
```

This will create a new `finetune_qwen_softmoe_logs.ipynb` with your changes.

## ⚖️ License

This notebook is provided for educational and research purposes. Please cite the original papers and respect the licenses of:
- Qwen2.5 model (Apache 2.0)
- Your dataset license
- Libraries used

## 📧 Support

For issues with:
- **The notebook**: Check the troubleshooting section above
- **Qwen model**: See [Qwen GitHub](https://github.com/QwenLM/Qwen)
- **Transformers**: See [Hugging Face docs](https://huggingface.co/docs)
- **Dataset**: Check your dataset's documentation

---

**Created for 7030CEM - Log Understanding and Analysis**

**Last Updated**: November 2024

**Version**: 1.0.0

