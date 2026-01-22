#!/usr/bin/env python3
"""
End-to-end validation script for Mimic-Video.
Uses tiny random weights and synthetic data for fast validation.
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    print("=" * 60)
    print("Mimic-Video End-to-End Validation")
    print("=" * 60)

    # Configuration
    # Note: batch_size=1 for sampling (model limitation during inference)
    config = {
        "batch_size": 1,
        "num_frames": 3,
        "height": 32,
        "width": 32,
        "action_chunk_len": 32,
        "dim_action": 20,
        "dim_joint_state": 32,
        "sample_steps": 16,
        "model_dim": 512,
        "depth": 3,
        "extract_layer": 1,
    }

    print(f"\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # Device selection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Import modules
    print("\n[1/5] Importing modules...")
    from mimic_video import MimicVideo
    from mimic_video.cosmos_predict import CosmosPredictWrapper
    print("  ✓ Imports successful")

    # Create video wrapper with tiny random weights
    print("\n[2/5] Creating CosmosPredictWrapper (tiny, random weights)...")
    video_wrapper = CosmosPredictWrapper(
        extract_layer=config["extract_layer"],
        random_weights=True,
        tiny=True,
    )
    video_wrapper = video_wrapper.to(device)
    print(f"  ✓ Video wrapper created")
    print(f"  ✓ Latent dimension: {video_wrapper.dim_latent}")

    # Create MimicVideo model
    print("\n[3/5] Creating MimicVideo model...")
    model = MimicVideo(
        dim=config["model_dim"],
        video_predict_wrapper=video_wrapper,
        action_chunk_len=config["action_chunk_len"],
        dim_action=config["dim_action"],
        dim_joint_state=config["dim_joint_state"],
        depth=config["depth"],
    )
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✓ Model created")
    print(f"  ✓ Total parameters: {total_params:,}")
    print(f"  ✓ Trainable parameters: {trainable_params:,}")

    # Create synthetic data
    print("\n[4/5] Creating synthetic data...")
    # Video: (batch, frames, channels, height, width) in [0, 1]
    video = torch.rand(
        config["batch_size"],
        config["num_frames"],
        3,
        config["height"],
        config["width"],
        device=device
    )

    # Joint state: (batch, dim_joint_state)
    joint_state = torch.randn(
        config["batch_size"],
        config["dim_joint_state"],
        device=device
    )

    # Prompts (single prompt for batch_size=1)
    prompts = "pick up the red cube"

    print(f"  ✓ Video shape: {video.shape}")
    print(f"  ✓ Joint state shape: {joint_state.shape}")
    print(f"  ✓ Prompt: {prompts}")

    # Run inference (sampling)
    print("\n[5/5] Running inference (sampling)...")
    print(f"  Steps: {config['sample_steps']}")

    model.eval()
    with torch.no_grad():
        # Sample actions
        predicted_actions = model.sample(
            video=video,
            joint_state=joint_state,
            prompts=prompts,
            steps=config["sample_steps"],
            disable_progress_bar=False,
        )

    print(f"\n  ✓ Predicted actions shape: {predicted_actions.shape}")
    print(f"  ✓ Expected shape: ({config['batch_size']}, {config['action_chunk_len']}, {config['dim_action']})")

    # Validate output shape
    expected_shape = (config["batch_size"], config["action_chunk_len"], config["dim_action"])
    assert predicted_actions.shape == expected_shape, f"Shape mismatch: {predicted_actions.shape} != {expected_shape}"
    print("  ✓ Shape validation passed!")

    # Save results
    print("\n" + "=" * 60)
    print("Saving Results")
    print("=" * 60)

    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save predicted actions
    actions_file = output_dir / f"predicted_actions_{timestamp}.npy"
    np.save(actions_file, predicted_actions.cpu().numpy())
    print(f"  ✓ Actions saved to: {actions_file}")

    # Save config and summary
    summary = {
        "timestamp": timestamp,
        "config": config,
        "device": device,
        "output_shape": list(predicted_actions.shape),
        "action_stats": {
            "mean": float(predicted_actions.mean()),
            "std": float(predicted_actions.std()),
            "min": float(predicted_actions.min()),
            "max": float(predicted_actions.max()),
        },
        "model_params": {
            "total": total_params,
            "trainable": trainable_params,
        },
        "status": "SUCCESS"
    }

    summary_file = output_dir / f"validation_summary_{timestamp}.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  ✓ Summary saved to: {summary_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION SUCCESSFUL")
    print("=" * 60)
    print(f"\nAction Statistics:")
    print(f"  Mean: {summary['action_stats']['mean']:.4f}")
    print(f"  Std:  {summary['action_stats']['std']:.4f}")
    print(f"  Min:  {summary['action_stats']['min']:.4f}")
    print(f"  Max:  {summary['action_stats']['max']:.4f}")

    return 0


if __name__ == "__main__":
    exit(main())
