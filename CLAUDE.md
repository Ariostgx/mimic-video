# CLAUDE.md - Project Guidelines for Mimic-Video Diagnostics

Hi Claude, let's set up the environment for the Mimic-Video project. Please execute the following sequence:

1. **Fork & Clone**: Use GitHub CLI (`gh`) to fork the repository `lucidrains/mimic-video` to my personal account and clone it here.
   - Note: This is Phil Wang's implementation. It is likely cleaner than the official research code.

2. **Environment Isolation**:
   - Create a new conda environment named `mimic` (Python 3.10).

4. **First Commit**:
   - `git add CLAUDE.md`
   - `git commit -m "chore: setup mimic-video playground for efficiency analysis"`
   - `git push`

5. **Dependency Check**:
   - Since this is a lucidrains repo, check `setup.py` or `pyproject.toml`.
   - Install the package in editable mode if possible (`pip install -e .`).


## 1. Project Mission
We are verifying the "Temporal Redundancy" hypothesis using Mimic-Video.
**Goal:** Generate a plot: X-axis = Diffusion Steps, Y-axis = Action Prediction Error (MSE).

## 2. Workflow Priorities

### Phase 0: Conda Environment Setup (CRITICAL - DO THIS FIRST)
**Isolate the workspace.**
1. Create a new environment: `conda create -n mimic python=3.10 -y`
2. Activate it: `conda activate mimic`
3. **Install Dependencies:**
   - Base Install: `conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y`
   - Project Specifics: `pip install diffusers transformers accelerate matplotlib numpy`
   - Check `setup.py` or `requirements.txt` in the repo for any esoteric packages (like `decord` or `einops`) and install them.

### Phase 1: Probe Setup & Code Analysis
1. Locate the inference loop and the `action_decoder`.
2. Insert a "Probe Hook": Inside the loop, at steps [5, 10, 20, 30, 40, 50], extract the intermediate `latents`.

### Phase 2: Action Decoding & Measurement
1. Pass intermediate `latents` into the `action_decoder`.
2. Compare decoded action vs. ground truth (or Step 50 output).
3. Compute MSE (Mean Squared Error).

### Phase 3: Reporting
1. Save results to JSON.
2. Generate `action_convergence.png`.

## 3. Rules of Engagement
- **Do not retrain.** Inference probing only.
- **Dependency Logic:** If a specific library version causes errors (e.g., `diffusers` version mismatch), prioritize the version that works with the pre-trained weights, usually recent stable versions are fine.