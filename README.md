# EEG reconstruction pipeline

The essential code for the working pipeline:

1. Trained EEG-to-VGG bridge → 2,000-step DIP low-level images.
2. Perceptogram EEG → 1024-dimensional unCLIP predictions.
3. DIP initialization + predicted CLIP → unCLIP diffusion → low-level/final evaluation.

This is the **unCLIP branch**, using vision embeddings and an empty text prompt.
It does not use the separate Versatile Diffusion vision/text-token models.
The code is self-contained: no imports from the old experiment directories,
no notebook execution, training utilities, diagnostic experiments, or vendored frameworks.
It uses your existing Modal volume `eeg-rpa-data`, prepared data, and trained VGG weights.
It does not retrain the VGG bridge or prepare the raw EEG dataset.

## Run

From this folder, using the workspace virtual environment with Modal installed:

```bash
# Exact settings of the completed subject-01, 200-image run.
../.venv/bin/python run.py start --run sub01 --subjects 1

# All ten subjects, evaluated separately using the trained pooled VGG checkpoint.
../.venv/bin/python run.py start --run all10 --subjects all

# A chosen subset. Multi-subject selections also default to the pooled checkpoint.
../.venv/bin/python run.py start --run selected --subjects 1,3,7

# Explicitly use the pooled checkpoint for subject 1.
../.venv/bin/python run.py start --run pooled_sub01 --subjects 1 --vgg pooled

# Optional experiment: first ten test images, proper img2img strength .4, guidance 5.
../.venv/bin/python run.py start --run img2img10 --subjects 1 --images 10 \
  --sampler img2img --strength .4 --guidance 5
```

`start` deploys the current code and launches a detached workflow. Closing the
terminal does not stop it. Each run has its own output directory; existing
experiments and checkpoints are read only. Run names cannot be reused for
different subjects, weights, sampling settings, or code.

```bash
../.venv/bin/python run.py status --run sub01
../.venv/bin/python run.py logs --run sub01
../.venv/bin/python run.py download --run sub01
../.venv/bin/python run.py resume --run sub01
```

Logs use tqdm. Ctrl+C in `logs` detaches without stopping remote work.
`resume` keeps the original settings and continues committed stages/batches.
The log stream covers this app's runs; stage bars identify their subject.

Stages can also be run individually, with the same run name and settings:

```bash
../.venv/bin/python run.py start --run staged --subjects 1 --stage vgg
# After it finishes:
../.venv/bin/python run.py resume --run staged --stage clip
../.venv/bin/python run.py resume --run staged --stage diffusion
../.venv/bin/python run.py resume --run staged --stage evaluate
```

## Preserved behavior

| Stage | Recipe |
|---|---|
| VGG EEG | Existing 17-channel, 175-sample, 250-Hz prepared test EEG; 80 repetitions averaged |
| VGG inference | Original checkpoint architecture; bf16; query chunks of 1,024, preserving within-chunk self-attention |
| DIP | VGG16-BN, five feature layers, grouped DIP network; batch 16; Adam 0.01; 2,000 steps; seed 42; original noise, AMP, loss and EMA operation order |
| CLIP EEG | Original Perceptogram 17-channel × 80-sample, 100-Hz EEG; training 4-repetition and test 80-repetition averages |
| CLIP targets | Pinned ViT-H/14 unCLIP encoder; 1,024 coordinates from the released noise-level-0 encoding; fp16 extraction, float64 storage |
| CLIP regression | Per-subject Ridge alpha 0, intercept enabled; training EEG z-score with ddof 1; coordinate calibration to training-target mean/std |
| Default diffusion | Released init: `0.02 * VAE(DIP).mode() + noise`; RGB [0,1] VAE input; 20 nominal steps; guidance 7.5; seed 0 per image; one output/image |
| Optional img2img | RGB [-1,1], scaled posterior mode; scheduler `add_noise`; suffix of denoising timesteps; default strength .4, guidance 5 |
| Evaluation | Same production-notebook metric functions, models and preprocessing; PixCorr, 425px Gaussian SSIM, AlexNet(2/5), Inception, CLIP, EffNet-B, SwAV and image/brain retrieval |

Python/Torch/library versions are separated into the original VGG, CLIP and
evaluation profiles. The decoder archive is pinned to revision
`ac4aaff4a67d57bdab2e9faf277c33d8e49f4df4`. Historical equality with the
unavailable original Stability AI repository is not independently established.

VGG and CLIP predictions/DIP caches cover **all 200 images**, even with
`--images 10`. This preserves DIP batch grouping and Perceptogram's calibration
across the full predicted test set. Only diffusion/evaluation is shortened.
Calibration uses predicted vectors, never true test embeddings. These remain
official test subsets, not training-derived validation samples.

Completed original VGG DIP caches and CLIP target caches are reused after
identity/checksum checks. Otherwise the same computations run here. There is
no averaging of raw EEG, predicted VGG tokens, or generated images across people.
The pooled VGG model was trained on all ten subjects; it is not a LOSO model.

## Outputs and prerequisites

Remote: `/mnt/eeg-rpa/eeg_pipeline/<run>/` on `eeg-rpa-data`.

```text
run.json, progress.json
sub-01/
  vgg/lowlevel.pt, recipe.json, complete.json
  clip/predictions.npy, weights.npz, recipe.json, complete.json
  diffusion/00000.png ... 00199.png, recipe.json, complete.json
  evaluation/metrics.json, metrics.csv, preview_000.png ...
per_subject.csv, aggregate.csv, summary.json
```

Each preview shows **ground truth | VGG DIP | final prediction**. Subject scores
are saved immediately. Aggregate tables use equal subject weighting and sample
SD; SD is empty for one subject. Retrieval has N candidates (200 by default),
and feature identification also depends on the selected image set. Compare
scores only on matched image sets.

`download` saves JSON, CSV, previews and final PNGs under `results/<run>/`;
large intermediate tensors and tokens stay remote. The original volume must
contain `prepared/sub-XX/test.npy` and its metadata, `images.json` and the stimulus
files, Perceptogram's prepared `perceptogram_vd_release_54db0981/data/`, and the
trained VGG paths listed in `common.py`. CLIP targets are extracted if their
verified existing cache is absent. Missing inputs fail explicitly.

## Files

- `run.py`: Modal environments, subject/stage selection, workflow, reports and downloads.
- `vgg.py`, `vgg_model.py`: checkpoint-compatible prediction and DIP.
- `clip.py`: target extraction and the released reconstruction regression.
- `diffusion.py`: both samplers.
- `evaluate.py`, `metric_functions.py`: original metrics and preview galleries.
- `common.py`: settings, paths, cache checks and aggregate calculations.
- Three requirements files and `model_assets.json`: pinned execution profiles/assets.

The source lineage is recorded in `source_manifest.json`. Perceptogram's license
is retained in `LICENSE_PERCEPTOGRAM`.

Verification used real GPU inputs: both VGG checkpoints matched saved predictions
bit for bit; a two-step DIP inversion matched the original; eight freshly encoded
CLIP targets matched cached targets exactly; regression outputs matched after
fp16 conversion; and two PNGs from each sampler matched the earlier runs byte
for byte. A two-image subject-01 workflow completed through aggregate evaluation
and download. These are sampled parity checks, not a full ten-subject rerun.
# eeg_decoding
