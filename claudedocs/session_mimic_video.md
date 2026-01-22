# Session: mimic_video
**Date**: 2026-01-21
**Project**: Mimic-Video Temporal Redundancy Analysis

---

## Session Overview

Setup and validation of the Mimic-Video codebase for temporal redundancy hypothesis testing on the SOTA cluster.

## Accomplishments

### 1. Environment Setup
- **Repository**: Forked `lucidrains/mimic-video` → `Ariostgx/mimic-video`
- **Conda Environment**: `mimic` (Python 3.10)
- **PyTorch**: 2.6.0+cu124
- **Dependencies**: diffusers, transformers, accelerate, einops, einx

### 2. Code Validation
- Created `scripts/validate_e2e.py` for end-to-end testing
- Verified model creation (30M params in tiny config)
- Confirmed action sampling works with synthetic data

### 3. GPU Cluster Integration
- Created Slurm scripts: `run_gpu_test.sh`, `run_probing.sh`
- **Critical Fix**: Added HuggingFace offline mode for worker nodes
  ```bash
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
  ```
- Pre-downloaded T5 tokenizer on head node

### 4. Bug Fixes
- `cosmos_predict.py`: Fixed tokenizer output device placement (CPU → GPU)
- `mimic_video.py`: Fixed `time_video_denoise` device placement

### 5. Temporal Redundancy Probing
- Created `scripts/probe_temporal_redundancy.py`
- Captures intermediate actions at steps [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
- Computes MSE vs final prediction
- Generates convergence plot

## Key Results

| Step | MSE vs Final |
|------|-------------|
| 5    | 0.2627      |
| 10   | 0.2093      |
| 15   | 0.1614      |
| 20   | 0.1194      |
| 25   | 0.0834      |
| 30   | 0.0536      |
| 35   | 0.0303      |
| 40   | 0.0135      |
| 45   | 0.0034      |
| 50   | 0.0000      |

**90% Convergence**: Step 40 (out of 50)

## File Structure

```
mimic-video/
├── CLAUDE.md                    # Project guidelines
├── claudedocs/
│   └── session_mimic_video.md   # This file
├── scripts/
│   ├── validate_e2e.py          # E2E validation
│   ├── probe_temporal_redundancy.py  # Probing script
│   ├── run_gpu_test.sh          # Slurm GPU test
│   └── run_probing.sh           # Slurm probing job
├── outputs/
│   ├── action_convergence_*.png # Convergence plot
│   ├── probe_results_*.json     # Full results
│   └── probe_actions_*.npz      # Raw action data
├── mimic_video/
│   ├── cosmos_predict.py        # Fixed device placement
│   └── mimic_video.py           # Fixed device placement
└── logs/                        # Slurm job logs
```

## Technical Notes

### Worker Node Limitations
- No internet access on sota-[1-8]
- Must pre-download HuggingFace models on head node
- Set offline environment variables in Slurm scripts

### Model Configuration (Tiny)
- `batch_size`: 1 (model limitation during inference)
- `model_dim`: 512
- `depth`: 3
- `extract_layer`: 1
- Total params: ~30M

### Commands to Resume

```bash
# Activate environment
source /u/shuhan/anaconda3/etc/profile.d/conda.sh
conda activate mimic
cd /u/shuhan/cc_work/mimic_video/mimic-video

# Run probing on GPU
sbatch scripts/run_probing.sh

# Check job status
squeue -u $USER

# View results
ls outputs/
```

## Next Steps

1. **Pretrained Model**: Load NVIDIA Cosmos 7B backbone
2. **Real Data**: Test with robot manipulation videos
3. **Deeper Analysis**: More probe steps, analyze per-dimension
4. **Early Stopping**: Investigate stopping at step 40 vs 50

## Git Commits

1. `chore: setup mimic-video playground for efficiency analysis`
2. `feat: add end-to-end validation script`
3. `feat: add temporal redundancy probing with GPU support`
