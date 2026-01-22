#!/bin/bash
#SBATCH --job-name=mimic_probe
#SBATCH --partition=rtxa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/probe_%j.out
#SBATCH --error=logs/probe_%j.err

# Create log directory
mkdir -p logs

# Setup environment
source /u/shuhan/anaconda3/etc/profile.d/conda.sh
conda activate mimic

# IMPORTANT: Set offline mode for HuggingFace (worker nodes have no internet)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Print job info
echo "=============================================="
echo "Mimic-Video Temporal Redundancy Probing"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"
echo ""

# Verify GPU
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Run probing script
echo "Running temporal redundancy probing..."
cd /u/shuhan/cc_work/mimic_video/mimic-video
python scripts/probe_temporal_redundancy.py

echo ""
echo "End: $(date)"
