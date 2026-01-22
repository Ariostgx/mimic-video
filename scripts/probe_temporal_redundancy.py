#!/usr/bin/env python3
"""
Temporal Redundancy Probing for Mimic-Video.

Goal: Verify the hypothesis that action predictions converge early in the diffusion process.
Generates: action_convergence.png showing MSE vs diffusion steps.

This script:
1. Runs diffusion sampling with intermediate state capture
2. Computes action predictions at each diffusion step
3. Measures MSE between intermediate and final predictions
4. Generates convergence plot
"""

import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def sample_with_probing(
    model,
    video,
    joint_state,
    prompts,
    total_steps: int = 50,
    probe_steps: list = None,
    device: str = "cpu",
):
    """
    Modified sampling that captures intermediate action predictions.

    Args:
        model: MimicVideo model
        video: Input video tensor
        joint_state: Joint state tensor
        prompts: Text prompts
        total_steps: Total diffusion steps
        probe_steps: Steps at which to capture intermediate actions
        device: Device to run on

    Returns:
        final_actions: Final predicted actions
        intermediate_actions: Dict mapping step -> action predictions
        intermediate_denoised: Dict mapping step -> denoised latents
    """
    if probe_steps is None:
        probe_steps = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    model.eval()

    batch_size = video.shape[0]
    action_shape = model.action_shape

    # Storage for intermediate results
    intermediate_actions = {}
    intermediate_denoised = {}

    with torch.no_grad():
        # Initialize times and noise
        times = torch.linspace(0., 1., total_steps + 1, device=device)[:-1]
        delta = 1. / total_steps

        # Start with pure noise
        noise = torch.randn((batch_size, *action_shape), device=device)
        denoised = noise.clone()

        cache = None

        # Get video features once (they're cached after first forward pass)
        # We need to do an initial forward to populate the cache

        print(f"  Running {total_steps} diffusion steps...")

        for step_idx, time in enumerate(tqdm(times, desc="  Diffusion", leave=False)):
            current_step = step_idx + 1

            # Forward pass
            # Note: prompts must always be passed when video_predict_wrapper exists
            # video is only needed on first iteration; cache handles subsequent iterations
            pred_flow, cache = model.forward(
                actions=denoised,
                time=time,
                cache=cache,
                return_cache=True,
                video=video if cache is None else None,
                joint_state=joint_state,
                prompts=prompts,
            )

            # Update denoised
            denoised = denoised + delta * pred_flow

            # Capture at probe steps
            if current_step in probe_steps:
                # Store the current denoised state
                intermediate_denoised[current_step] = denoised.clone()

                # The denoised tensor IS the action prediction at this step
                # (before inverse normalization)
                actions_at_step = denoised.clone()

                # Apply inverse normalization if model has normalizer
                if model.action_normalizer is not None:
                    actions_at_step = model.action_normalizer.inverse_normalize(actions_at_step)

                intermediate_actions[current_step] = actions_at_step

        # Final actions
        final_actions = denoised.clone()
        if model.action_normalizer is not None:
            final_actions = model.action_normalizer.inverse_normalize(final_actions)

    return final_actions, intermediate_actions, intermediate_denoised


def compute_mse_convergence(intermediate_actions: dict, final_actions: torch.Tensor):
    """
    Compute MSE between intermediate predictions and final prediction.

    Returns:
        steps: List of step numbers
        mse_values: List of MSE values
    """
    steps = sorted(intermediate_actions.keys())
    mse_values = []

    for step in steps:
        intermediate = intermediate_actions[step]
        mse = torch.nn.functional.mse_loss(intermediate, final_actions).item()
        mse_values.append(mse)

    return steps, mse_values


def plot_convergence(steps, mse_values, output_path, config):
    """Generate the action convergence plot."""

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(steps, mse_values, 'b-o', linewidth=2, markersize=8, label='MSE vs Final')

    # Mark key regions
    ax.axhline(y=mse_values[-1] * 1.1, color='g', linestyle='--', alpha=0.5, label='10% of final MSE')

    ax.set_xlabel('Diffusion Step', fontsize=12)
    ax.set_ylabel('MSE (vs Final Prediction)', fontsize=12)
    ax.set_title('Temporal Redundancy Analysis: Action Prediction Convergence', fontsize=14)

    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add text box with config
    textstr = f"Total Steps: {config['total_steps']}\nModel Dim: {config['model_dim']}\nAction Dim: {config['dim_action']}"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"  ✓ Plot saved to: {output_path}")


def main():
    print("=" * 60)
    print("Temporal Redundancy Probing - Mimic-Video")
    print("=" * 60)

    # Configuration
    config = {
        "batch_size": 1,
        "num_frames": 3,
        "height": 32,
        "width": 32,
        "action_chunk_len": 32,
        "dim_action": 20,
        "dim_joint_state": 32,
        "total_steps": 50,  # More steps for better resolution
        "probe_steps": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        "model_dim": 512,
        "depth": 3,
        "extract_layer": 1,
    }

    print(f"\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Import and create models
    print("\n[1/5] Setting up models...")
    from mimic_video import MimicVideo
    from mimic_video.cosmos_predict import CosmosPredictWrapper

    video_wrapper = CosmosPredictWrapper(
        extract_layer=config["extract_layer"],
        random_weights=True,
        tiny=True,
    ).to(device)

    model = MimicVideo(
        dim=config["model_dim"],
        video_predict_wrapper=video_wrapper,
        action_chunk_len=config["action_chunk_len"],
        dim_action=config["dim_action"],
        dim_joint_state=config["dim_joint_state"],
        depth=config["depth"],
    ).to(device)

    print(f"  ✓ Models created on {device}")

    # Create synthetic data
    print("\n[2/5] Creating synthetic data...")
    video = torch.rand(
        config["batch_size"],
        config["num_frames"],
        3,
        config["height"],
        config["width"],
        device=device
    )
    joint_state = torch.randn(config["batch_size"], config["dim_joint_state"], device=device)
    prompts = "pick up the red cube and place it on the table"

    print(f"  ✓ Video: {video.shape}")
    print(f"  ✓ Joint state: {joint_state.shape}")

    # Run probing
    print("\n[3/5] Running diffusion with probing...")
    final_actions, intermediate_actions, _ = sample_with_probing(
        model=model,
        video=video,
        joint_state=joint_state,
        prompts=prompts,
        total_steps=config["total_steps"],
        probe_steps=config["probe_steps"],
        device=device,
    )

    print(f"  ✓ Final actions shape: {final_actions.shape}")
    print(f"  ✓ Captured {len(intermediate_actions)} intermediate states")

    # Compute MSE convergence
    print("\n[4/5] Computing MSE convergence...")
    steps, mse_values = compute_mse_convergence(intermediate_actions, final_actions)

    print(f"\n  Step -> MSE:")
    for step, mse in zip(steps, mse_values):
        print(f"    {step:3d}: {mse:.6f}")

    # Generate plot
    print("\n[5/5] Generating convergence plot...")
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = output_dir / f"action_convergence_{timestamp}.png"

    plot_convergence(steps, mse_values, plot_path, config)

    # Save results to JSON
    results = {
        "timestamp": timestamp,
        "config": config,
        "device": device,
        "convergence_data": {
            "steps": steps,
            "mse_values": mse_values,
        },
        "final_action_stats": {
            "mean": float(final_actions.mean()),
            "std": float(final_actions.std()),
            "min": float(final_actions.min()),
            "max": float(final_actions.max()),
        },
        "convergence_analysis": {
            "initial_mse": mse_values[0],
            "final_mse": mse_values[-1],
            "mse_reduction_ratio": mse_values[0] / max(mse_values[-1], 1e-10),
            "step_90_percent_converged": None,  # Will compute below
        }
    }

    # Find step at which 90% convergence is achieved
    threshold = mse_values[0] * 0.1  # 90% reduction
    for i, (step, mse) in enumerate(zip(steps, mse_values)):
        if mse <= threshold:
            results["convergence_analysis"]["step_90_percent_converged"] = step
            break

    results_path = output_dir / f"probe_results_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ Results saved to: {results_path}")

    # Save intermediate actions as numpy
    actions_data = {
        "final": final_actions.cpu().numpy(),
        "intermediate": {str(k): v.cpu().numpy() for k, v in intermediate_actions.items()}
    }
    np.savez(output_dir / f"probe_actions_{timestamp}.npz", **{
        "final": actions_data["final"],
        **{f"step_{k}": v for k, v in actions_data["intermediate"].items()}
    })
    print(f"  ✓ Actions saved to: {output_dir}/probe_actions_{timestamp}.npz")

    # Print summary
    print("\n" + "=" * 60)
    print("PROBING COMPLETE")
    print("=" * 60)
    print(f"\nKey Findings:")
    print(f"  Initial MSE (step {steps[0]}): {mse_values[0]:.6f}")
    print(f"  Final MSE (step {steps[-1]}):   {mse_values[-1]:.6f}")
    print(f"  MSE Reduction Ratio: {results['convergence_analysis']['mse_reduction_ratio']:.2f}x")

    if results["convergence_analysis"]["step_90_percent_converged"]:
        print(f"  90% Convergence at Step: {results['convergence_analysis']['step_90_percent_converged']}")
    else:
        print(f"  90% Convergence: Not achieved within probe steps")

    print(f"\nOutput files:")
    print(f"  - {plot_path}")
    print(f"  - {results_path}")

    return 0


if __name__ == "__main__":
    exit(main())
