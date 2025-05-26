# Unified LTXV Training System

This guide explains how to use the streamlined training system that handles preprocessing, training, and checkpoint conversion through a single configuration file.

## Quick Start

1. **Setup Environment** (first time only):
   ```bash
   # Windows
   setup.bat
   
   # Linux/macOS
   source venv/bin/activate
   ./setup.sh
   ```

2. **Configure Training**:
   Edit `configs/unified_training_config.yaml` to match your dataset and training preferences.

3. **Run Training**:
   ```bash
   # Windows
   train.bat configs/unified_training_config.yaml
   
   # Linux/macOS
   ./train.sh configs/unified_training_config.yaml
   
   # Or directly with Python
   python train.py configs/unified_training_config.yaml
   ```

## Configuration Guide

The unified config file (`configs/unified_training_config.yaml`) controls all aspects of the training pipeline:

### Dataset Configuration

```yaml
dataset:
  # Path to folder containing paired video and text files
  # Each video file should have a corresponding text file with the same name
  data_folder: "data/videos"
  
  # Resolution and frame buckets
  resolution_buckets:
    - [768, 768, 97]  # [width, height, frames]
    - [512, 512, 49]  # Can specify multiple buckets
  
  # Optional trigger word for LoRA training
  id_token: "SQUISH"  # Prepended to all captions
```

**Important**: If you change the `resolution_buckets` settings, the system will automatically detect this and reprocess your dataset. This ensures that the preprocessed data always matches your current training settings.

### Training Configuration

```yaml
training:
  # Single GPU or distributed training
  distributed: false
  num_processes: null  # Auto-detect GPUs if null
  
  # Training parameters
  learning_rate: 2e-4
  steps: 1500
  batch_size: 1
  gradient_accumulation_steps: 1
```

### Model Configuration

```yaml
model:
  model_source: "LTXV_13B_097_DEV"
  training_mode: "lora"  # "lora" or "full"

lora:  # Only used if training_mode is "lora"
  rank: 128
  alpha: 128
  dropout: 0.0
```

### Automatic Features

The unified system automatically:

1. **Smart preprocessing**: Skips preprocessing if valid precomputed data exists, otherwise encodes videos and text embeddings
2. **Handles distributed training**: Uses accelerate for multi-GPU training
3. **Converts checkpoints**: Automatically converts LoRA checkpoints to ComfyUI format after training completes
4. **Organizes outputs**: Creates timestamped directories for each training run
5. **Reuses preprocessed data**: Shares preprocessed data across multiple training runs for efficiency

## Dataset Format

The unified system expects a simple folder structure with paired video and text files:

```
data/videos/
├── video1.mp4       # Video file
├── video1.txt       # Caption for video1.mp4
├── video2.mov       # Another video file
├── video2.txt       # Caption for video2.mov
├── scene3.avi       # Video file
├── scene3.txt       # Caption for scene3.avi
└── ...
```

**Requirements:**
- Each video file must have a corresponding `.txt` file with the same name
- Text files should contain a single line with the caption for that video
- Supported video formats: `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`, `.m4v`
- Video and text files must be in the same directory

**Example text file content:**
```
# video1.txt
A person walking through a beautiful park on a sunny day
```

## Advanced Usage

### Distributed Training

Set `distributed: true` in your config to enable multi-GPU training:

```yaml
training:
  distributed: true
  num_processes: 4  # Or null for auto-detection
```

### Custom Preprocessing

Adjust preprocessing settings:

```yaml
dataset:
  preprocessing:
    batch_size: 2
    num_workers: 4
    vae_tiling: true  # For large videos
    decode_videos: true  # Save decoded videos for verification
```

### Checkpoint Management

Control checkpoint behavior:

```yaml
checkpoints:
  interval: 250  # Save every 250 steps
  keep_last_n: 3  # Keep only last 3 checkpoints
  auto_convert_to_comfy: true  # Auto-convert to ComfyUI format
```

**Auto-Conversion Features:**
- **Post-training conversion**: Checkpoints are converted to ComfyUI format after training completes
- **Robust detection**: Converts all `.safetensors` files (not just those with "lora" in the name)
- **Skip duplicates**: Won't re-convert files that already have ComfyUI versions
- **No performance impact**: Conversion happens after training, so it doesn't slow down the training process

## Output Structure

The training pipeline creates timestamped output directories for better organization:

```
outputs/
├── run_20250526_143022/   # Timestamped training run
│   ├── checkpoints/       # Training checkpoints
│   ├── validation/        # Validation videos (if enabled)
│   └── logs/              # Training logs
├── run_20250526_151045/   # Another training run
│   └── ...
└── ...

data/videos/
└── .precomputed/          # Preprocessed data (shared across runs)
    ├── latents/           # Video latents
    └── conditions/        # Text embeddings
```

**Key Features:**
- **Timestamped runs**: Each training run gets its own folder with date/time
- **Shared preprocessing**: Preprocessed data is reused across multiple training runs
- **Automatic organization**: No need to manually manage output directories

## Troubleshooting

### Common Issues

1. **Module not found errors**: Run `setup.bat` or `setup.sh` to install dependencies
2. **CUDA out of memory**: Reduce `batch_size` or enable `gradient_checkpointing`
3. **Video encoding errors**: Check video file formats and paths
4. **Distributed training fails**: Ensure all GPUs are available and accelerate is configured
5. **Training hangs during validation**: Set `validation.interval: null` to disable validation
6. **Validation takes too long**: Use smaller `video_dims` like `[256, 256, 25]` and fewer `inference_steps`

### Performance Issues

**If training is very slow:**
- Disable validation: `validation.interval: null`
- Reduce batch size if using gradient accumulation
- Enable mixed precision: `mixed_precision_mode: "bf16"`
- Use gradient checkpointing: `enable_gradient_checkpointing: true`

**If validation hangs:**
- The default validation settings generate large videos (704x480x161 frames)
- This can take 10+ minutes per validation video
- Either disable validation or use much smaller dimensions

**Preprocessing Optimization:**
- **First run**: Preprocessing will encode all videos and text (can take time)
- **Subsequent runs**: Automatically detects and reuses existing preprocessed data
- **Resolution validation**: Checks that precomputed data matches current resolution bucket settings
- **Automatic reprocessing**: Forces reprocessing if resolution buckets change
- **Manual cleanup**: Delete `data/videos/.precomputed/` to force re-preprocessing
- **Metadata tracking**: Saves preprocessing settings to prevent mismatched data reuse

### Performance Tips

- **Disable validation initially**: Set `validation.interval: null` for faster training
- **Use smaller validation videos**: If validation is enabled, use small dimensions like `[256, 256, 25]`
- **Reduce validation steps**: Use `inference_steps: 20` instead of 50 for validation
- Use `vae_tiling: true` for large resolution videos
- Enable `gradient_checkpointing` to save memory
- Use `mixed_precision_mode: "bf16"` for faster training
- Set `num_workers: 0` if you encounter data loading issues

### Validation Settings

Validation can significantly slow down training as it generates sample videos. Consider these settings:

```yaml
validation:
  interval: null  # Disable validation completely
  # OR for occasional validation:
  interval: 1000  # Only validate every 1000 steps
  video_dims: [256, 256, 25]  # Small videos for speed
  inference_steps: 20  # Fewer steps for speed
```

## Migration from Old Scripts

If you were using the individual scripts before:

**Old way:**
```bash
python scripts/preprocess_dataset.py data/dataset.json --resolution-buckets "768x768x97"
python scripts/train.py configs/config.yaml
python scripts/convert_checkpoint.py checkpoint.safetensors --to-comfy
```

**New way:**
```bash
python train.py configs/unified_training_config.yaml
```

The unified system handles all these steps automatically based on your configuration.