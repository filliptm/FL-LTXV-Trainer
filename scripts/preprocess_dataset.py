#!/usr/bin/env python3

"""
Preprocess a video dataset by computing video clips latents and text captions embeddings.

This script provides a command-line interface for preprocessing video datasets by computing
latent representations of video clips and text embeddings of their captions. The preprocessed
data can be used to accelerate training of video generation models and to save GPU memory.

Basic usage:
    preprocess_dataset.py /path/to/dataset --resolution-buckets 768x768x49

The dataset can be either:
1. A directory containing text files with captions and video paths
2. A CSV, JSON, or JSONL file with columns for captions and video paths
"""

from fractions import Fraction
from pathlib import Path
from typing import Any

import torch
import torchvision
import typer
from pydantic import BaseModel
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from torch.utils.data import DataLoader
from transformers.utils.logging import disable_progress_bar

from ltxv_trainer.datasets import (
    PRECOMPUTED_CONDITIONS_DIR_NAME,
    PRECOMPUTED_LATENTS_DIR_NAME,
    ImageOrVideoDatasetWithResizeAndRectangleCrop,
)
from ltxv_trainer.ltxv_utils import decode_video, encode_prompt, encode_video
from ltxv_trainer.model_loader import LtxvModelVersion, load_text_encoder, load_tokenizer, load_vae

disable_progress_bar()
console = Console()
app = typer.Typer(
    pretty_exceptions_enable=False,
    help="Preprocess a video dataset by computing video clips latents and text captions embeddings. "
    "The dataset can be either a directory with text files or a CSV/JSON/JSONL file.",
)

VAE_SPATIAL_FACTOR = 32
VAE_TEMPORAL_FACTOR = 8


class PreprocessingArgs(BaseModel):
    """Arguments for dataset preprocessing"""

    dataset_path: str
    caption_column: str
    video_column: str
    resolution_buckets: list[tuple[int, int, int]]
    batch_size: int
    num_workers: int
    output_dir: str | None
    id_token: str | None
    vae_tiling: bool
    decode_videos: bool


class DatasetPreprocessor:
    def __init__(self, model_source: str, device: str = "cuda", load_text_encoder_in_8bit: bool = False):
        """Initialize the preprocessor with model configuration.

        Args:
            model_source: Model source - can be a version string (e.g. "LTXV_2B_0.9.5"), HF repo, or local path
            device: Device to use for computation
            load_text_encoder_in_8bit: Whether to load text encoder in 8-bit precision
        """
        self.device = torch.device(device)
        self._load_models(model_source, load_text_encoder_in_8bit)

    @torch.inference_mode()
    def preprocess(self, args: PreprocessingArgs) -> None:  # noqa: PLR0912
        """Run the preprocessing pipeline with the given arguments"""
        console.print("[bold blue]Starting preprocessing...[/]")

        # Determine if dataset_path is a file or directory
        dataset_path = Path(args.dataset_path)
        is_file = dataset_path.is_file()

        # Set data_root and dataset_file based on dataset_path
        if is_file:
            data_root = str(dataset_path.parent)
            dataset_file = str(dataset_path)

            # Validate that the file exists
            if not dataset_path.exists():
                raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

            # Validate file type
            if dataset_path.suffix.lower() not in [".csv", ".json", ".jsonl"]:
                raise ValueError(f"Dataset file must be CSV, JSON, or JSONL format: {dataset_path}")
        else:
            data_root = str(dataset_path)
            dataset_file = None

            # Validate that the directory exists
            if not dataset_path.exists() or not dataset_path.is_dir():
                raise FileNotFoundError(f"Dataset path does not exist: {dataset_path} or is not a directory")

            # Check for required files if dataset_path is a directory
            caption_file = Path(data_root) / args.caption_column
            video_file = Path(data_root) / args.video_column

            # Add .txt extension if needed
            if not caption_file.suffix:
                caption_file = caption_file.with_suffix(".txt")
                args.caption_column += ".txt"

            if not video_file.suffix:
                video_file = video_file.with_suffix(".txt")
                args.video_column += ".txt"

            # Check if caption file exists
            if not caption_file.exists():
                raise FileNotFoundError(f"Captions file does not exist: {caption_file}")

            # Check if video file exists
            if not video_file.exists():
                raise FileNotFoundError(f"Video paths file does not exist: {video_file}")

        # Set up output directories
        output_base = Path(args.output_dir) if args.output_dir else Path(data_root) / ".precomputed"
        latents_dir, conditions_dir = self._create_output_dirs(output_base)

        if args.id_token:
            console.print(
                f"[bold yellow]LoRA trigger word[/] [cyan]{args.id_token}[/] "
                f"[bold yellow]will be prepended to all captions[/]",
            )

        # Set up data loading
        dataloader = self._create_dataloader(
            data_root=data_root,
            dataset_file=dataset_file,
            caption_column=args.caption_column,
            video_column=args.video_column,
            resolution_buckets=args.resolution_buckets,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            id_token=args.id_token,
        )

        # Enable/disable VAE tiling
        if args.vae_tiling:
            self.vae.enable_tiling()
        else:
            self.vae.disable_tiling()

        # Print dataset information
        console.print(f"Number of batches: {len(dataloader)} (batch size: {args.batch_size})")

        # Create progress bars
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console,
        )

        with progress:
            # Process batches
            task = progress.add_task(
                "Processing dataset",
                total=len(dataloader),
            )

            for batch in dataloader:
                self._process_batch(
                    batch=batch,
                    latents_dir=latents_dir,
                    conditions_dir=conditions_dir,
                    output_base=output_base,
                    decode_videos=args.decode_videos,
                )
                progress.advance(task)

        # Print summary
        console.print(
            f"[bold green]✓[/] Processed [bold]{len(dataloader.dataset)}[/] items. "
            f"Results saved to [cyan]{output_base}[/]",
        )

    def _load_models(self, model_source: str, load_text_encoder_in_8bit: bool) -> None:
        """Initialize and load the required models"""
        with console.status(f"[bold]Loading models from [cyan]{model_source}[/]...", spinner="dots"):
            # Load only the components we need for preprocessing
            self.vae = load_vae(model_source, dtype=torch.bfloat16).to(self.device)
            self.tokenizer = load_tokenizer()
            self.text_encoder = load_text_encoder(load_in_8bit=load_text_encoder_in_8bit).to(self.device)

        console.print("[bold green]✓[/] Models loaded successfully")

    @staticmethod
    def _create_output_dirs(output_base: Path) -> tuple[Path, Path]:
        """Create and return paths for output directories"""
        latents_dir = output_base / PRECOMPUTED_LATENTS_DIR_NAME
        conditions_dir = output_base / PRECOMPUTED_CONDITIONS_DIR_NAME

        latents_dir.mkdir(parents=True, exist_ok=True)
        conditions_dir.mkdir(parents=True, exist_ok=True)

        return latents_dir, conditions_dir

    @staticmethod
    def _create_dataloader(
        data_root: str,
        dataset_file: str | None,
        caption_column: str,
        video_column: str,
        resolution_buckets: list[tuple[int, int, int]],
        batch_size: int,
        num_workers: int,
        id_token: str | None,
    ) -> DataLoader:
        """Initialize dataset and create dataloader"""
        with console.status("[bold]Loading dataset...", spinner="dots"):
            dataset = ImageOrVideoDatasetWithResizeAndRectangleCrop(
                data_root=data_root,
                dataset_file=dataset_file,
                caption_column=caption_column,
                video_column=video_column,
                resolution_buckets=resolution_buckets,
                id_token=id_token,
            )

        console.print(f"[bold green]✓[/] Dataset loaded with [bold]{len(dataset)}[/] items")

        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )

    def _process_batch(
        self,
        batch: dict[str, Any],
        latents_dir: Path,
        conditions_dir: Path,
        output_base: Path,
        decode_videos: bool,
    ) -> None:
        """Process a single batch of data and save the results"""
        # Encode prompts
        text_embeddings = encode_prompt(
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            prompt=batch["prompt"],
            device=self.device,
        )

        # Encode videos/images
        video_latents = encode_video(
            vae=self.vae,
            image_or_video=batch["video"],
            device=self.device,
        )

        # Save each item in the batch
        for i in range(len(batch["prompt"])):
            # Get relative path and create output directories maintaining structure
            rel_path = Path(batch["relative_paths"][i])
            rel_dir = rel_path.parent

            # Create output directories
            latent_subdir = latents_dir / rel_dir
            condition_subdir = conditions_dir / rel_dir
            latent_subdir.mkdir(parents=True, exist_ok=True)
            condition_subdir.mkdir(parents=True, exist_ok=True)

            # Save files with original name
            latent_path = latent_subdir / f"{rel_path.stem}.pt"
            condition_path = condition_subdir / f"{rel_path.stem}.pt"

            fps = batch["video_metadata"]["fps"][i].item()
            latent_item = {
                "latents": video_latents["latents"][i].cpu().contiguous(),
                "num_frames": video_latents["num_frames"],
                "height": video_latents["height"],
                "width": video_latents["width"],
                "fps": fps,
            }
            condition_item = {
                "prompt_embeds": text_embeddings["prompt_embeds"][i].cpu().contiguous(),
                "prompt_attention_mask": text_embeddings["prompt_attention_mask"][i].cpu().contiguous(),
            }

            torch.save(latent_item, latent_path)
            torch.save(condition_item, condition_path)

            # Decode video/image if requested
            if decode_videos:
                decoded_dir = output_base / "decoded_videos" / rel_dir
                decoded_dir.mkdir(parents=True, exist_ok=True)

                video = decode_video(
                    vae=self.vae,
                    latents=latent_item["latents"],
                    num_frames=latent_item["num_frames"],
                    height=latent_item["height"],
                    width=latent_item["width"],
                    device=self.device,
                )
                video = video[0]  # Remove batch dimension
                # Convert to uint8 for saving
                video = (video * 255).round().clamp(0, 255).to(torch.uint8)
                video = video.permute(1, 2, 3, 0)  # [C,F,H,W] -> [F,H,W,C]

                # For single frame (images), save as PNG, otherwise as MP4
                is_image = video.shape[0] == 1
                if is_image:
                    output_path = decoded_dir / f"{rel_path.stem}.png"
                    torchvision.utils.save_image(
                        video[0].permute(2, 0, 1) / 255.0,  # [H,W,C] -> [C,H,W] and normalize
                        str(output_path),
                    )
                else:
                    output_path = decoded_dir / f"{rel_path.stem}.mp4"
                    torchvision.io.write_video(
                        str(output_path),
                        video.cpu(),
                        fps=Fraction(fps).limit_denominator(1000),
                        video_codec="h264",
                        options={"crf": "18"},
                    )


def _parse_resolution_buckets(resolution_buckets_str: str) -> list[tuple[int, int, int]]:
    """Parse resolution buckets from string format to list of tuples"""
    resolution_buckets = []
    for bucket_str in resolution_buckets_str.split(";"):
        w, h, f = map(int, bucket_str.split("x"))

        if w % VAE_SPATIAL_FACTOR != 0 or h % VAE_SPATIAL_FACTOR != 0:
            raise typer.BadParameter(
                f"Width and height must be multiples of {VAE_SPATIAL_FACTOR}, got {w}x{h}",
                param_hint="resolution-buckets",
            )

        if f % VAE_TEMPORAL_FACTOR != 1:
            raise typer.BadParameter(
                f"Number of frames must be a multiple of {VAE_TEMPORAL_FACTOR} plus 1, got {f}",
                param_hint="resolution-buckets",
            )

        resolution_buckets.append((f, h, w))
    return resolution_buckets


@app.command()
def main(  # noqa: PLR0913
    dataset_path: str = typer.Argument(
        ...,
        help="Path to dataset directory or metadata file (CSV/JSON/JSONL)",
    ),
    resolution_buckets: str = typer.Option(
        ...,
        help='Resolution buckets in format "WxHxF;WxHxF;..." (e.g. "768x768x25;512x512x49")',
    ),
    caption_column: str = typer.Option(
        default="caption",
        help="Column name or filename for captions: "
        "If dataset_path is a CSV/JSON/JSONL file, this is the column name containing captions. "
        "If dataset_path is a directory, this is the filename containing line-separated captions.",
    ),
    video_column: str = typer.Option(
        default="media_path",
        help="Column name or filename for videos: "
        "If dataset_path is a CSV/JSON/JSONL file, this is the column name containing video paths. "
        "If dataset_path is a directory, this is the filename containing line-separated video paths.",
    ),
    batch_size: int = typer.Option(
        default=1,
        help="Batch size for preprocessing",
    ),
    num_workers: int = typer.Option(
        default=1,
        help="Number of dataloader workers",
    ),
    device: str = typer.Option(
        default="cuda",
        help="Device to use for computation",
    ),
    load_text_encoder_in_8bit: bool = typer.Option(
        default=False,
        help="Load the T5 text encoder in 8-bit precision to save memory",
    ),
    vae_tiling: bool = typer.Option(
        default=False,
        help="Enable VAE tiling for larger video resolutions",
    ),
    output_dir: str | None = typer.Option(
        default=None,
        help="Output directory (defaults to .precomputed in dataset directory)",
    ),
    model_source: str = typer.Option(
        default=str(LtxvModelVersion.latest()),
        help="Model source - can be a version string (e.g. 'LTXV_2B_0.9.5'), HF repo, or local path",
    ),
    id_token: str | None = typer.Option(
        default=None,
        help="Optional token to prepend to each caption (acts as a trigger word when training a LoRA)",
    ),
    decode_videos: bool = typer.Option(
        default=False,
        help="Decode and save videos after encoding (for verification purposes)",
    ),
) -> None:
    """Preprocess a video dataset by computing and saving latents and text embeddings.

    The dataset can be specified in two ways:
    1. A directory containing text files with captions and video paths
    2. A CSV, JSON, or JSONL file with columns for captions and video paths
    """
    parsed_resolution_buckets = _parse_resolution_buckets(resolution_buckets)

    if len(parsed_resolution_buckets) > 1:
        raise typer.BadParameter(
            "Multiple resolution buckets are not yet supported. Please specify only one bucket.",
            param_hint="resolution-buckets",
        )

    args = PreprocessingArgs(
        dataset_path=dataset_path,
        caption_column=caption_column,
        video_column=video_column,
        resolution_buckets=parsed_resolution_buckets,
        batch_size=batch_size,
        num_workers=num_workers,
        output_dir=output_dir,
        id_token=id_token,
        vae_tiling=vae_tiling,
        decode_videos=decode_videos,
    )

    preprocessor = DatasetPreprocessor(
        model_source=model_source,
        device=device,
        load_text_encoder_in_8bit=load_text_encoder_in_8bit,
    )
    preprocessor.preprocess(args)


if __name__ == "__main__":
    app()
