#!/usr/bin/env python3

"""
Unified LTXV Training Pipeline

This script provides a complete training pipeline that handles:
1. Dataset preprocessing (video encoding and text embedding)
2. Model training (single GPU or distributed)
3. Automatic checkpoint conversion to ComfyUI format

All configuration is controlled through a single YAML file.

Basic usage:
    python scripts/unified_train.py configs/unified_training_config.yaml
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import torch
import typer
import yaml
from pydantic import BaseModel, Field
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from ltxv_trainer.config import LtxvTrainerConfig
from ltxv_trainer.trainer import LtxvTrainer
from ltxv_trainer.utils import convert_checkpoint

console = Console()
app = typer.Typer(
    pretty_exceptions_enable=False,
    help="Unified LTXV training pipeline with preprocessing, training, and conversion.",
)


class DatasetConfig(BaseModel):
    """Dataset configuration"""
    data_folder: str
    resolution_buckets: list[list[int]]
    preprocessing: Dict[str, Any] = Field(default_factory=dict)
    id_token: str | None = None


class TrainingConfig(BaseModel):
    """Training configuration"""
    distributed: bool = False
    num_processes: int | None = None
    learning_rate: float
    steps: int
    batch_size: int
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    optimizer_type: str = "adamw"
    scheduler_type: str = "linear"
    scheduler_params: Dict[str, Any] = Field(default_factory=dict)
    enable_gradient_checkpointing: bool = True
    first_frame_conditioning_p: float = 0.5


class CheckpointConfig(BaseModel):
    """Checkpoint configuration"""
    interval: int | None = 250
    keep_last_n: int = -1
    auto_convert_to_comfy: bool = True


class UnifiedConfig(BaseModel):
    """Unified training configuration"""
    dataset: DatasetConfig
    model: Dict[str, Any]
    lora: Dict[str, Any] = Field(default_factory=dict)
    training: TrainingConfig
    acceleration: Dict[str, Any] = Field(default_factory=dict)
    validation: Dict[str, Any] = Field(default_factory=dict)
    checkpoints: CheckpointConfig
    flow_matching: Dict[str, Any] = Field(default_factory=dict)
    hub: Dict[str, Any] = Field(default_factory=dict)
    seed: int = 42
    output_dir: str


class UnifiedTrainer:
    def __init__(self, config: UnifiedConfig):
        self.config = config
        self.console = console
        
    def run_pipeline(self) -> None:
        """Run the complete training pipeline"""
        self.console.print("[bold blue]🚀 Starting Unified LTXV Training Pipeline[/]")
        
        # Create timestamped output directory
        self._setup_timestamped_output()
        
        # Step 1: Check for existing preprocessed data or preprocess dataset
        preprocessed_data_path = self._handle_preprocessing()
        
        # Step 2: Update config with preprocessed data path
        self._update_config_for_training(preprocessed_data_path)
        
        # Step 3: Run training
        self._run_training()
        
        # Step 4: Convert checkpoints to ComfyUI format
        if self.config.checkpoints.auto_convert_to_comfy:
            self._convert_checkpoints()
            
        self.console.print(f"[bold green]✅ Pipeline completed successfully![/]")
        self.console.print(f"[bold blue]📁 Results saved to: {self.config.output_dir}[/]")

    def _setup_timestamped_output(self) -> None:
        """Create a timestamped output directory for this training run"""
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get base output directory from config
        base_output = Path(self.config.output_dir)
        
        # Create timestamped subdirectory
        timestamped_output = base_output / f"run_{timestamp}"
        timestamped_output.mkdir(parents=True, exist_ok=True)
        
        # Update config with new output directory
        self.config.output_dir = str(timestamped_output)
        
        self.console.print(f"[blue]📁 Training run output directory: {timestamped_output}[/]")

    def _handle_preprocessing(self) -> str:
        """Handle preprocessing - skip if precomputed data exists, otherwise preprocess"""
        dataset_config = self.config.dataset
        data_folder = Path(dataset_config.data_folder)
        
        # Check for existing precomputed data
        precomputed_path = data_folder / ".precomputed"
        if precomputed_path.exists() and self._validate_precomputed_data(precomputed_path):
            self.console.print("[bold green]✅ Found existing precomputed data, skipping preprocessing[/]")
            self.console.print(f"[blue]📂 Using precomputed data from: {precomputed_path}[/]")
            return str(precomputed_path)
        else:
            self.console.print("[bold yellow]📊 No valid precomputed data found, starting preprocessing...[/]")
            return self._preprocess_dataset()

    def _validate_precomputed_data(self, precomputed_path: Path) -> bool:
        """Validate that precomputed data is complete and matches current resolution settings"""
        try:
            # Check for required subdirectories
            latents_dir = precomputed_path / "latents"
            conditions_dir = precomputed_path / "conditions"
            
            if not latents_dir.exists() or not conditions_dir.exists():
                self.console.print("[yellow]⚠️ Precomputed data missing required directories[/]")
                return False
            
            # Check if directories have content
            latent_files = list(latents_dir.glob("**/*.pt"))
            condition_files = list(conditions_dir.glob("**/*.pt"))
            
            if not latent_files or not condition_files:
                self.console.print("[yellow]⚠️ Precomputed data directories are empty[/]")
                return False
            
            # Check if counts match
            if len(latent_files) != len(condition_files):
                self.console.print("[yellow]⚠️ Mismatch between latent and condition file counts[/]")
                return False
            
            # Check resolution bucket compatibility
            if not self._validate_resolution_compatibility(precomputed_path, latent_files):
                return False
            
            self.console.print(f"[green]✅ Validated {len(latent_files)} precomputed samples[/]")
            return True
            
        except Exception as e:
            self.console.print(f"[yellow]⚠️ Error validating precomputed data: {e}[/]")
            return False

    def _validate_resolution_compatibility(self, precomputed_path: Path, latent_files: list) -> bool:
        """Check if precomputed data matches current resolution bucket settings"""
        try:
            # Load metadata file if it exists
            metadata_file = precomputed_path / "preprocessing_metadata.yaml"
            current_buckets = self.config.dataset.resolution_buckets
            
            if metadata_file.exists():
                # Compare with saved metadata
                import yaml
                with open(metadata_file, 'r') as f:
                    saved_metadata = yaml.safe_load(f)
                
                saved_buckets = saved_metadata.get('resolution_buckets', [])
                if saved_buckets != current_buckets:
                    self.console.print(f"[yellow]⚠️ Resolution bucket mismatch![/]")
                    self.console.print(f"[yellow]   Saved: {saved_buckets}[/]")
                    self.console.print(f"[yellow]   Current: {current_buckets}[/]")
                    return False
            else:
                # No metadata file - check actual latent dimensions
                self.console.print("[blue]📋 No metadata file found, checking latent dimensions...[/]")
                
                # Sample a few latent files to check dimensions
                sample_files = latent_files[:min(3, len(latent_files))]
                for latent_file in sample_files:
                    try:
                        import torch
                        latent_data = torch.load(latent_file, map_location='cpu')
                        
                        # Get dimensions from latent data
                        height = latent_data.get('height')
                        width = latent_data.get('width')
                        num_frames = latent_data.get('num_frames')
                        
                        if height and width and num_frames:
                            # Check if this matches any of our current buckets
                            found_match = False
                            for bucket in current_buckets:
                                bucket_w, bucket_h, bucket_f = bucket
                                if width == bucket_w and height == bucket_h and num_frames == bucket_f:
                                    found_match = True
                                    break
                            
                            if not found_match:
                                self.console.print(f"[yellow]⚠️ Found precomputed data with dimensions {width}x{height}x{num_frames}[/]")
                                self.console.print(f"[yellow]   This doesn't match current buckets: {current_buckets}[/]")
                                return False
                    except Exception as e:
                        self.console.print(f"[yellow]⚠️ Error reading latent file {latent_file}: {e}[/]")
                        return False
                
                # Save current metadata for future checks
                self._save_preprocessing_metadata(precomputed_path)
            
            return True
            
        except Exception as e:
            self.console.print(f"[yellow]⚠️ Error validating resolution compatibility: {e}[/]")
            return False

    def _save_preprocessing_metadata(self, precomputed_path: Path) -> None:
        """Save preprocessing metadata for future validation"""
        try:
            import yaml
            metadata = {
                'resolution_buckets': self.config.dataset.resolution_buckets,
                'preprocessing_timestamp': datetime.now().isoformat(),
                'id_token': self.config.dataset.id_token,
            }
            
            metadata_file = precomputed_path / "preprocessing_metadata.yaml"
            with open(metadata_file, 'w') as f:
                yaml.dump(metadata, f, default_flow_style=False)
                
        except Exception as e:
            self.console.print(f"[yellow]⚠️ Warning: Could not save preprocessing metadata: {e}[/]")

    def _preprocess_dataset(self) -> str:
        """Preprocess the dataset and return the path to preprocessed data"""
        self.console.print("[bold yellow]📊 Step 1: Preprocessing Dataset[/]")
        
        # Import preprocessing components
        from scripts.preprocess_dataset import DatasetPreprocessor, PreprocessingArgs
        
        dataset_config = self.config.dataset
        preprocessing_config = dataset_config.preprocessing
        
        # Convert resolution buckets format
        resolution_buckets = []
        for bucket in dataset_config.resolution_buckets:
            if len(bucket) != 3:
                raise ValueError(f"Resolution bucket must have 3 values [width, height, frames], got {bucket}")
            w, h, f = bucket
            resolution_buckets.append((f, h, w))  # Convert to (frames, height, width)
        
        # Create temporary caption and video files for the preprocessor
        data_folder = Path(dataset_config.data_folder)
        if not data_folder.exists():
            raise FileNotFoundError(f"Data folder does not exist: {data_folder}")
            
        # Find all video files and their corresponding text files
        video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'}
        video_files = []
        captions = []
        skipped_videos = 0
        skipped_texts = 0
        
        # First pass: collect all video files that have corresponding text files
        all_files = list(data_folder.iterdir())
        video_file_paths = [f for f in all_files if f.suffix.lower() in video_extensions]
        text_file_paths = [f for f in all_files if f.suffix.lower() == '.txt']
        
        self.console.print(f"[blue]Found {len(video_file_paths)} video files and {len(text_file_paths)} text files[/]")
        
        for video_file in video_file_paths:
            # Look for corresponding text file
            text_file = video_file.with_suffix('.txt')
            if text_file.exists() and text_file in text_file_paths:
                try:
                    with open(text_file, 'r', encoding='utf-8') as f:
                        caption = f.read().strip()
                    
                    # Skip empty captions
                    if not caption:
                        self.console.print(f"[yellow]Warning: Empty caption for {video_file.name}, skipping[/]")
                        skipped_videos += 1
                        continue
                    
                    # Add id_token if specified
                    if dataset_config.id_token:
                        caption = f"{dataset_config.id_token} {caption}"
                    
                    video_files.append(str(video_file.relative_to(data_folder)))
                    captions.append(caption)
                except Exception as e:
                    self.console.print(f"[yellow]Warning: Error reading {text_file.name}: {e}, skipping[/]")
                    skipped_videos += 1
            else:
                self.console.print(f"[yellow]Warning: No text file found for {video_file.name}[/]")
                skipped_videos += 1
        
        # Check for orphaned text files (text files without corresponding videos)
        for text_file in text_file_paths:
            video_file = text_file.with_suffix('.mp4')  # Check common extensions
            if not video_file.exists():
                # Try other extensions
                found_video = False
                for ext in video_extensions:
                    potential_video = text_file.with_suffix(ext)
                    if potential_video.exists():
                        found_video = True
                        break
                if not found_video:
                    skipped_texts += 1
        
        if skipped_videos > 0:
            self.console.print(f"[yellow]Skipped {skipped_videos} videos (missing/empty text files)[/]")
        if skipped_texts > 0:
            self.console.print(f"[yellow]Found {skipped_texts} orphaned text files (no corresponding video)[/]")
        
        if not video_files:
            raise ValueError(f"No valid paired video and text files found in {data_folder}")
            
        self.console.print(f"[green]Found {len(video_files)} valid paired video/text files[/]")
        
        # Verify counts match
        if len(video_files) != len(captions):
            raise ValueError(f"Internal error: video files ({len(video_files)}) and captions ({len(captions)}) count mismatch")
        
        # Create temporary files for the preprocessor
        temp_caption_file = data_folder / "temp_captions.txt"
        temp_video_file = data_folder / "temp_videos.txt"
        
        try:
            # Write temporary files
            with open(temp_caption_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(captions))
            
            with open(temp_video_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(video_files))
            
            # Set up preprocessing arguments
            preprocessing_args = PreprocessingArgs(
                dataset_path=str(data_folder),
                caption_column="temp_captions.txt",
                video_column="temp_videos.txt",
                resolution_buckets=resolution_buckets,
                batch_size=preprocessing_config.get("batch_size", 1),
                num_workers=preprocessing_config.get("num_workers", 0),
                output_dir=preprocessing_config.get("output_dir"),
                id_token=None,  # Already handled above
                vae_tiling=preprocessing_config.get("vae_tiling", False),
                decode_videos=preprocessing_config.get("decode_videos", False),
            )
            
            # Initialize preprocessor
            preprocessor = DatasetPreprocessor(
                model_source=self.config.model["model_source"],
                device="cuda",  # Default to cuda
                load_text_encoder_in_8bit=preprocessing_config.get("load_text_encoder_in_8bit", False),
            )
            
            # Run preprocessing
            preprocessor.preprocess(preprocessing_args)
            
        finally:
            # Clean up temporary files
            if temp_caption_file.exists():
                temp_caption_file.unlink()
            if temp_video_file.exists():
                temp_video_file.unlink()
        
        # Save preprocessing metadata and return the preprocessed data path
        output_path = preprocessing_args.output_dir if preprocessing_args.output_dir else str(data_folder / ".precomputed")
        self._save_preprocessing_metadata(Path(output_path))
        
        return output_path

    def _update_config_for_training(self, preprocessed_data_path: str) -> None:
        """Update the configuration to use preprocessed data for training"""
        # Create a training config that uses the preprocessed data
        # Remove device from model config as it's not a valid field in LtxvTrainerConfig
        model_config = dict(self.config.model)
        model_config.pop("device", None)  # Remove device if present
        
        self.training_config_dict = {
            "model": model_config,
            "lora": self.config.lora,
            "optimization": {
                "learning_rate": self.config.training.learning_rate,
                "steps": self.config.training.steps,
                "batch_size": self.config.training.batch_size,
                "gradient_accumulation_steps": self.config.training.gradient_accumulation_steps,
                "max_grad_norm": self.config.training.max_grad_norm,
                "optimizer_type": self.config.training.optimizer_type,
                "scheduler_type": self.config.training.scheduler_type,
                "scheduler_params": self.config.training.scheduler_params,
                "enable_gradient_checkpointing": self.config.training.enable_gradient_checkpointing,
                "first_frame_conditioning_p": self.config.training.first_frame_conditioning_p,
            },
            "acceleration": self.config.acceleration,
            "data": {
                "preprocessed_data_root": preprocessed_data_path,
                "num_dataloader_workers": self.config.dataset.preprocessing.get("num_workers", 2),
            },
            "validation": self.config.validation,
            "checkpoints": {
                "interval": self.config.checkpoints.interval,
                "keep_last_n": self.config.checkpoints.keep_last_n,
            },
            "flow_matching": self.config.flow_matching,
            "hub": self.config.hub,
            "seed": self.config.seed,
            "output_dir": self.config.output_dir,
        }

    def _run_training(self) -> None:
        """Run the training process"""
        self.console.print("[bold yellow]🏋️ Step 2: Training Model[/]")
        
        if self.config.training.distributed:
            self._run_distributed_training()
        else:
            self._run_single_gpu_training()

    def _run_single_gpu_training(self) -> None:
        """Run single GPU training"""
        try:
            trainer_config = LtxvTrainerConfig(**self.training_config_dict)
            trainer = LtxvTrainer(trainer_config)
            trainer.train()
        except Exception as e:
            self.console.print(f"[bold red]❌ Training failed: {e}[/]")
            raise

    def _run_distributed_training(self) -> None:
        """Run distributed training using accelerate"""
        self.console.print("[bold blue]🔄 Launching distributed training...[/]")
        
        # Save temporary config file for distributed training
        temp_config_path = Path(self.config.output_dir) / "temp_training_config.yaml"
        temp_config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(temp_config_path, "w") as f:
            yaml.dump(self.training_config_dict, f, default_flow_style=False)
        
        # Determine number of processes
        num_processes = self.config.training.num_processes
        if num_processes is None:
            try:
                gpu_list = subprocess.check_output(["nvidia-smi", "-L"], encoding="utf-8")
                num_processes = len(gpu_list.split("\n")) - 1
                self.console.print(f"[blue]Auto-detected {num_processes} GPUs[/]")
            except (subprocess.CalledProcessError, FileNotFoundError):
                self.console.print("[yellow]Failed to detect GPUs, using 1 process[/]")
                num_processes = 1
        
        # Build accelerate command
        script_dir = Path(__file__).parent
        training_script = str(script_dir / "train.py")
        
        cmd = [
            "accelerate", "launch",
            "--num_processes", str(num_processes),
        ]
        
        if num_processes > 1:
            cmd.append("--multi_gpu")
            
        cmd.extend([training_script, str(temp_config_path)])
        
        # Run distributed training
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            self.console.print("[green]✅ Distributed training completed[/]")
        except subprocess.CalledProcessError as e:
            self.console.print(f"[bold red]❌ Distributed training failed: {e}[/]")
            self.console.print(f"[red]stdout: {e.stdout}[/]")
            self.console.print(f"[red]stderr: {e.stderr}[/]")
            raise
        finally:
            # Clean up temporary config file
            if temp_config_path.exists():
                temp_config_path.unlink()

    def _convert_checkpoints(self) -> None:
        """Convert LoRA checkpoints to ComfyUI format"""
        self.console.print("[bold yellow]🔄 Step 3: Converting Checkpoints to ComfyUI Format[/]")
        
        output_dir = Path(self.config.output_dir)
        if not output_dir.exists():
            self.console.print("[yellow]No output directory found, skipping conversion[/]")
            return
            
        # Find all .safetensors checkpoint files (more robust detection)
        checkpoint_files = list(output_dir.glob("**/*.safetensors"))
        
        # Filter out already converted files and other non-checkpoint files
        lora_checkpoints = []
        for f in checkpoint_files:
            # Skip if already a comfy version
            if "_comfy" in f.name.lower():
                continue
            # Skip if it's a VAE or other model file
            if any(skip in f.name.lower() for skip in ["vae", "text_encoder", "unet"]):
                continue
            # Include all other .safetensors files (likely checkpoints)
            lora_checkpoints.append(f)
        
        if not lora_checkpoints:
            self.console.print("[yellow]No checkpoint files found for conversion[/]")
            return
            
        self.console.print(f"[blue]Found {len(lora_checkpoints)} checkpoint files to convert[/]")
            
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Converting checkpoints...", total=len(lora_checkpoints))
            
            for checkpoint_path in lora_checkpoints:
                # Skip if already converted
                comfy_path = checkpoint_path.parent / f"{checkpoint_path.stem}_comfy{checkpoint_path.suffix}"
                if comfy_path.exists():
                    self.console.print(f"[yellow]⏭️ Already converted: {checkpoint_path.name}[/]")
                    progress.advance(task)
                    continue
                    
                try:
                    convert_checkpoint(str(checkpoint_path), str(comfy_path), to_comfy=True)
                    self.console.print(f"[green]✅ Converted: {checkpoint_path.name} -> {comfy_path.name}[/]")
                except Exception as e:
                    self.console.print(f"[red]❌ Failed to convert {checkpoint_path.name}: {e}[/]")
                    
                progress.advance(task)
    


@app.command()
def main(config_path: str = typer.Argument(..., help="Path to unified training configuration YAML file")) -> None:
    """Run the unified LTXV training pipeline."""
    
    # Load configuration
    config_path = Path(config_path)
    if not config_path.exists():
        console.print(f"[bold red]❌ Configuration file not found: {config_path}[/]")
        raise typer.Exit(code=1)
        
    try:
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)
        config = UnifiedConfig(**config_data)
    except Exception as e:
        console.print(f"[bold red]❌ Invalid configuration: {e}[/]")
        raise typer.Exit(code=1)
    
    # Create output directory
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run the pipeline
    trainer = UnifiedTrainer(config)
    try:
        trainer.run_pipeline()
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️ Training interrupted by user[/]")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"\n[bold red]❌ Pipeline failed: {e}[/]")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()