"""Perceptogram's 1024-dimensional unCLIP targets and reconstruction regression."""

from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge

from common import (
    DATA,
    EEG_CLIP,
    MODEL,
    OLD_TARGETS,
    check_order,
    guard,
    read_json,
    sha256,
    write_json,
)


def encode_image(encoder, processor, normalizer, scheduler, image):
    """The released image encoder followed by noise-level-0 normalization/noising."""
    import torch
    from diffusers.utils.torch_utils import randn_tensor

    with torch.no_grad():
        pixels = processor(images=image, return_tensors="pt").pixel_values.cuda().half()
        clean = encoder(pixels).image_embeds
        noise = randn_tensor(clean.shape, device=clean.device, dtype=clean.dtype)
        released = scheduler.add_noise(
            normalizer.scale(clean), noise=noise, timesteps=torch.tensor([0], device="cuda")
        )
        return normalizer.unscale(released)[0].double().cpu().numpy()


def targets(commit=lambda: None):
    """Reuse verified original targets, or extract them one image at a time in fp16."""
    import torch
    from PIL import Image
    from tqdm.auto import tqdm
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    from diffusers import DDPMScheduler
    from diffusers.pipelines.stable_diffusion.stable_unclip_image_normalizer import (
        StableUnCLIPImageNormalizer,
    )
    from huggingface_hub import hf_hub_download

    check_order()
    destination = DATA / "eeg_pipeline/shared/clip_targets"
    destination.mkdir(parents=True, exist_ok=True)
    original = (
        read_json(OLD_TARGETS / "complete.json")
        if (OLD_TARGETS / "complete.json").exists()
        else None
    )
    if original:
        specification = original["specification"]
        metadata = EEG_CLIP / "thingseeg2_metadata/image_metadata.npy"
        if (
            specification["model_assets"] != MODEL
            or sha256(metadata) != specification["input_sha256"][metadata.name]
        ):
            raise ValueError("Original targets use different assets or stimulus metadata")
    # Cached targets already record the full stimulus checksums. Avoid rescanning
    # the 12-GB training-image array when no images are being encoded.
    stimuli = {
        s: original["specification"]["input_sha256"][f"{s}_images.npy"]
        if original
        else sha256(EEG_CLIP / f"thingseeg2_metadata/{s}_images.npy")
        for s in ("train", "test")
    }
    identity = {
        "repository": MODEL["repository"],
        "revision": MODEL["revision"],
        "batch_size": 1,
        "dtype": "float16 extraction, float64 storage",
        "seed_per_split": 0,
        "definition": "released _encode_image first 1024 coordinates at noise_level=0",
        "stimuli": stimuli,
    }
    guard(destination / "recipe.json", identity)
    marker = destination / "complete.json"
    if marker.exists():
        saved = read_json(marker)
        for path, expected in saved["sha256"].items():
            if sha256(path) != expected:
                raise ValueError("CLIP target checksum mismatch")
        return saved
    if original:
        if read_json(OLD_TARGETS / "revision.json") != {
            "repository": MODEL["repository"],
            "revision": MODEL["revision"],
        }:
            raise ValueError("Original target model revision differs")
        paths = {s: OLD_TARGETS / f"{s}_release_noise0_1024.npy" for s in ("train", "test")}
        for path in paths.values():
            if sha256(path) != original["files_sha256"][path.name]:
                raise ValueError("Original CLIP target checksum mismatch")
    else:
        # Only the encoder components are needed here, not the diffusion weights.
        snapshot = None
        for name, spec in MODEL["files"].items():
            path = Path(
                hf_hub_download(MODEL["repository"], filename=name, revision=MODEL["revision"])
            )
            if path.stat().st_size != spec["size"] or (
                "sha256" in spec and sha256(path) != spec["sha256"]
            ):
                raise ValueError(f"Pinned encoder asset changed: {name}")
            if "sha256" not in spec:
                import hashlib

                data = path.read_bytes()
                blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
                if blob != spec["git_blob_sha1"]:
                    raise ValueError(f"Pinned encoder configuration changed: {name}")
            if snapshot is not None and path.parent.parent != snapshot:
                raise ValueError("Mixed encoder snapshots")
            snapshot = path.parent.parent
        encoder = (
            CLIPVisionModelWithProjection.from_pretrained(
                snapshot,
                subfolder="image_encoder",
                torch_dtype=torch.float16,
                use_safetensors=False,
            )
            .cuda()
            .eval()
            .requires_grad_(False)
        )
        processor = CLIPImageProcessor.from_pretrained(snapshot, subfolder="feature_extractor")
        normalizer = StableUnCLIPImageNormalizer.from_pretrained(
            snapshot, subfolder="image_normalizer", use_safetensors=False
        ).to(torch_device="cuda", torch_dtype=torch.float16)
        scheduler = DDPMScheduler.from_pretrained(snapshot, subfolder="image_noising_scheduler")
        paths = {}
        for split, count in [("train", 16540), ("test", 200)]:
            images = np.load(EEG_CLIP / f"thingseeg2_metadata/{split}_images.npy", mmap_mode="r")
            if images.shape != (count, 500, 500, 3) or images.dtype != np.uint8:
                raise ValueError("Unexpected original stimulus array")
            path = destination / f"{split}.npy"
            temporary = path.with_name("." + path.name)
            values = np.lib.format.open_memmap(
                temporary, mode="w+", dtype=np.float64, shape=(count, 1024)
            )
            torch.manual_seed(0)
            torch.cuda.manual_seed_all(0)
            for i in tqdm(range(count), desc=f"unCLIP targets {split}", unit="image"):
                image = Image.fromarray(images[i]).convert("RGB")
                values[i] = encode_image(encoder, processor, normalizer, scheduler, image)
            values.flush()
            del values
            temporary.replace(path)
            paths[split] = path
            commit()
    report = {
        "paths": {s: str(p) for s, p in paths.items()},
        "sha256": {str(p): sha256(p) for p in paths.values()},
    }
    write_json(marker, report)
    commit()
    return report


def fit(train, test, target):
    """Exactly the released alpha-0 Ridge with intercept and coordinate calibration."""
    train = train.astype(np.float64).reshape(len(train), -1)
    test = test.astype(np.float64).reshape(len(test), -1)
    mean, scale = train.mean(0), train.std(0, ddof=1)
    if (scale == 0).any():
        raise ValueError("Constant EEG coordinate in released regression")
    model = Ridge(alpha=0, max_iter=50000, fit_intercept=True).fit((train - mean) / scale, target)
    raw = model.predict((test - mean) / scale)
    spread = raw.std(0)
    if (spread == 0).any():
        raise ValueError("Undefined Perceptogram prediction calibration")
    prediction = (raw - raw.mean(0)) / spread * target.std(0) + target.mean(0)
    return prediction, model, mean, scale


def run(settings, subject, commit=lambda: None):
    cache = read_json(DATA / "eeg_pipeline/shared/clip_targets/complete.json")
    eeg = EEG_CLIP / f"thingseeg2_preproc/sub-{subject:02d}"
    output = settings.subject_dir(subject) / "clip"
    identity = {
        "subject": subject,
        "alpha": 0,
        "fit_intercept": True,
        "calibration_images": 200,
        "true_test_targets_used_for_fit": False,
        "train_targets_sha256": cache["sha256"][cache["paths"]["train"]],
        "eeg_sha256": {s: sha256(eeg / f"{s}_thingseeg2_avg.npy") for s in ("train", "test")},
        "implementation_sha256": sha256(__file__),
    }
    guard(output / "recipe.json", identity)
    if (output / "complete.json").exists():
        result = read_json(output / "complete.json")
        if sha256(output / "predictions.npy") != result["sha256"]:
            raise ValueError("CLIP prediction checksum mismatch")
        return result
    train, test = [np.load(eeg / f"{s}_thingseeg2_avg.npy") for s in ("train", "test")]
    target = np.load(cache["paths"]["train"])
    if (
        train.shape != (16540, 17, 80)
        or test.shape != (200, 17, 80)
        or target.shape != (16540, 1024)
    ):
        raise ValueError("Perceptogram requires original 100-Hz EEG and 1024-dimensional targets")
    if any(not np.isfinite(a).all() for a in (train, test, target)):
        raise ValueError("Nonfinite regression input")
    prediction, model, mean, scale = fit(train, test, target)
    if not np.isfinite(prediction).all():
        raise ValueError("Nonfinite CLIP predictions")
    np.save(output / "predictions.npy", prediction)
    np.savez(
        output / "weights.npz",
        weight=model.coef_,
        bias=model.intercept_,
        input_mean=mean,
        input_scale=scale,
    )
    result = {
        "subject": subject,
        "N": 200,
        "shape": [200, 1024],
        "sha256": sha256(output / "predictions.npy"),
    }
    write_json(output / "complete.json", result)
    commit()
    return result
