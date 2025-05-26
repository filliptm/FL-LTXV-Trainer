#!/usr/bin/env python3

"""
Simple launcher for the unified LTXV training pipeline.

This is a convenience script that launches the unified training pipeline.
You can use this directly or use scripts/unified_train.py for more control.

Usage:
    python train.py configs/unified_training_config.yaml
"""

import sys
from pathlib import Path

# Add the scripts directory to the path so we can import the unified trainer
scripts_dir = Path(__file__).parent / "scripts"
sys.path.insert(0, str(scripts_dir))

from unified_train import app

if __name__ == "__main__":
    app()