# Project Context: Gated Feature Fusion Bridge for MambaVision

Paste this into `.github/copilot-instructions.md` (or keep it as `PROJECT_CONTEXT.md` and reference it in Copilot Chat with `#file`).

## Goal
Computer Vision class project (about 6 weeks, small scale). Pick a paper, identify a gap, and improve on it with a controlled experiment. A small or negative result is acceptable if the setup is controlled and honestly reported.

## Base paper
**MambaVision** (NVIDIA, CVPR). Hierarchical hybrid backbone with 4 stages:
- Stages 1-2: CNN residual blocks (local features).
- Stages 3-4: Mamba mixer blocks, with self-attention blocks in the second half of each stage.
- Model variant: MambaVision-T. Stage 2 output is `[B, 160, 28, 28]`, Stage 3 input is `[B, 320, 14, 14]` (verify against the actual code and input size).

## Hypothesis (not an established fact)
The stride-2 downsample between Stage 2 and Stage 3 may limit how much local spatial detail Stage 3 and Stage 4 can use. Giving Stage 3 direct access to Stage 2 features may improve its features and the segmentation of object boundaries.

Note: UperNet already receives Stage 2 features directly, so the claim is about Stage 3/4 feature quality, not about the decoder losing information. Do not frame the gap as proven.

## Proposed change: Gated Feature Fusion Bridge
- Cache the Stage 2 output feature map.
- Downsample with a stride-2 conv to 14x14, then a 1x1 conv to 320 channels.
- Multiply by a learnable sigmoid gate and add as a residual to the Stage 3 input.
- **Initialization:** the model must start with baseline behavior. Initialize the gate strongly negative and/or zero-init the projection conv.
- Gate variants to ablate if time allows: scalar, per-channel, spatial/input-dependent, and an ungated bridge.
- Keep the bridge small and plug-and-play (a separate `nn.Module`, enabled by a config flag).

## Experimental design
- **Backbone init:** ImageNet-pretrained MambaVision weights for both models.
- **Decoder:** UperNet (mmsegmentation), randomly initialized head.
- **Dataset:** ADE20K, using a shortened schedule or subset so each run finishes in hours. Full 160k-iteration training is out of scope.
- **Training strategy (main experiment):** freeze Stages 1 and 2; train Stages 3 and 4, the bridge, and the head. Optionally use a lower backbone learning rate than the head.
- **Optional ablation:** fully frozen backbone (only head and bridge train). A null result there means "a frozen backbone can't use the bridge", not "the bridge doesn't work".
- **Controls:**
  - Baseline = identical config with no bridge (same freezing, schedule, data, augmentations, seed).
  - The bridge run starts from the same ImageNet weights, not from the fine-tuned baseline.
  - If time allows, add a parameter-matched control (e.g., a wider downsample conv) so gains can't be explained by extra parameters.
  - Use 2-3 seeds if compute allows; otherwise use the same seed for both and state that variance was not measured.

## Evaluation
- Overall mIoU and pixel accuracy.
- **Boundary-band analysis** (my own script, run on saved predictions):
  - Build a boundary mask from ground truth (pixels whose right or down neighbor has a different label), then take all pixels within `d` pixels of it using `scipy.ndimage.distance_transform_edt`, for d = 3, 5, 10.
  - Accumulate confusion matrices across the whole dataset (not per-image averages), separately for band and interior pixels, ignoring the ignore label.
  - Main number: pixel accuracy in band vs interior. Band mIoU is noisy for rare classes.
  - Key comparison: the change in band accuracy vs the change in interior accuracy between baseline and bridge.
  - Treat ignore regions as a label when building boundaries; use the same choice for both models and state it.
- Also report params, FLOPs, and inference throughput, plus qualitative before/after mask crops.
- Optional: Boundary IoU (Cheng et al., CVPR 2021).

## Sanity checks for the scoring pipeline
1. Band + interior pixel counts equal total valid pixels; combined accuracy matches mmseg's reported pixel accuracy.
2. Interior accuracy should be higher than band accuracy.
3. Band fraction should grow with `d`.
4. Visually check a saved boundary mask against an image.
5. Label indexing: ADE20K raw annotations use 0 = unlabeled, 1-150 = classes; mmseg `reduce_zero_label` maps to 0-149 with 255 = ignore. Prediction PNGs must be single-channel class IDs (not colored overlays), same size as GT, same indexing.
6. If a released ADE20K MambaVision checkpoint exists (not confirmed), use it only to validate the pipeline against published numbers, not as the baseline.

## Environment constraints
- Dev machine is a Mac (zsh); `mamba_ssm` and `causal-conv1d` CUDA kernels need an NVIDIA GPU, so training and inference run on Colab or a university cluster.
- The scoring script is plain NumPy/SciPy/PIL and can run locally on saved predictions.
- Check the installed mmsegmentation version's docs for how to dump prediction masks (`tools/test.py ... --out`), and write a custom inference loop saving `pred.astype(np.uint8)` PNGs if needed.

## Rough timeline
- Week 1: baseline training and evaluation working end to end; settle GPU access.
- Weeks 2-3: implement the bridge, train it.
- Week 4: one or two ablations at most (gated vs ungated, or gate type).
- Weeks 5-6: boundary analysis, figures, write-up.

## Guidance for Copilot
- Keep changes minimal and config-driven; the bridge must be switchable off so the baseline uses identical code paths.
- Don't invent API details for mmsegmentation or MambaVision; read the installed package source and say when unsure.
- Preserve reproducibility: set and log seeds, save configs alongside checkpoints, and log params/FLOPs/throughput.
- When unsure whether something would confound the baseline vs bridge comparison, flag it instead of assuming.
