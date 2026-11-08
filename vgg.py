"""Trained EEG-to-VGG inference and the original 2,000-step DIP inversion."""

from pathlib import Path
from types import SimpleNamespace
import gc
import random

import numpy as np
import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

from common import DATA, VGG_RUNS, check_order, guard, read_json, sha256, torch_save, write_json
from vgg_model import EEGVGG, BrainITVGGTokenizer, BatchedDIPUNet, tokens_to_brainit_vgg_features


def invert(features, tokenizer, generator, steps=2000):
    """Keep the notebook's RNG, AMP, loss accumulation, and EMA operation order."""
    count = len(features[0])
    model = BatchedDIPUNet(count).cuda().train()
    z = torch.randn(count, 32, 112, 112, generator=generator, device="cuda") * 0.1
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    targets = [layer.cuda().float().detach() for layer in features]
    ema = None
    progress = tqdm(
        range(steps), desc=f"DIP ({count} images)", unit="step", leave=False, mininterval=2
    )
    for step in progress:
        noise = torch.randn(z.shape, generator=generator, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            image = model(z + noise * (1.0 / 30.0))
            predicted = tokenizer.extract_features(image)
            loss = image.new_tensor(0.0)
            for p, t in zip(predicted, targets):
                loss = loss + 1.0 * F.mse_loss(p.float(), t.float())
            loss = loss / 5.0
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite DIP loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            current = model(z).float()
            ema = current if ema is None else 0.99 * ema + (1.0 - 0.99) * current
        if step % 20 == 0 or step == steps - 1:
            progress.set_postfix(loss=f"{float(loss.detach()):.4f}", refresh=False)
    return (
        F.interpolate(ema.clamp(0, 1), (256, 256), mode="bicubic", align_corners=False)
        .clamp(0, 1)
        .cpu()
    )


def checkpoint(profile):
    relative, expected = VGG_RUNS[profile]
    root = DATA / relative
    path = root / "best.pt"
    if sha256(path) != expected:
        raise ValueError(f"Unexpected trained VGG checkpoint: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    config = dict(
        patch_dim=256,
        patch_size=25,
        patch_stride=5,
        dropout=0.1,
        gradient_checkpointing=True,
        prediction_chunk=1024,
    )
    config.update(
        {k: v for k, v in read_json(root / "run.json").get("config", {}).items() if k in config}
    )
    model = EEGVGG(SimpleNamespace(**config)).cuda().eval()
    model.load_state_dict(state["model"], strict=True)
    return root, expected, model


def existing_shards(root, digest, subject, profile):
    folder = root / "evaluation" / digest
    if profile == "pooled":
        folder = folder / "subject_evaluations" / f"sub-{subject:02d}"
    return folder / "dip_n200_steps2000_batch16_seed42/notebook_dip_v1/shards"


def run(settings, subject, commit=lambda: None):
    check_order()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    root, digest, model = checkpoint(settings.vgg)
    output = settings.subject_dir(subject) / "vgg"
    source = DATA / "prepared" / f"sub-{subject:02d}/test.npy"
    metadata = read_json(source.with_suffix(".json"))
    if sha256(source) != metadata["sha256"] or metadata["repetitions_averaged"] != 80:
        raise ValueError("Prepared test EEG identity differs")
    channels = [
        "P7",
        "P5",
        "P3",
        "P1",
        "Pz",
        "P2",
        "P4",
        "P6",
        "P8",
        "PO7",
        "PO3",
        "POz",
        "PO4",
        "PO8",
        "O1",
        "Oz",
        "O2",
    ]
    if metadata["channels"] != channels or not np.allclose(
        metadata["time_seconds"], np.arange(175) * 0.004
    ):
        raise ValueError("VGG requires the original 17-channel 0..696ms input window")
    x = np.load(source, mmap_mode="r")
    if x.shape != (200, 17, 175) or x.dtype != np.float32 or not np.isfinite(x).all():
        raise ValueError("VGG requires prepared [200,17,175] float32 test EEG")
    config = vars(model.c)
    identity = {
        "subject": subject,
        "checkpoint": str(root / "best.pt"),
        "checkpoint_sha256": digest,
        "eeg_sha256": metadata["sha256"],
        "query_chunk": config["prediction_chunk"],
        "dip_steps": 2000,
        "dip_batch_size": 16,
        "dip_seed": 42,
        "implementation": {
            name: sha256(Path(__file__).with_name(name)) for name in ("vgg.py", "vgg_model.py")
        },
    }
    guard(output / "recipe.json", identity)
    if (output / "complete.json").exists():
        saved = read_json(output / "complete.json")
        if sha256(output / "lowlevel.pt") != saved["sha256"]:
            raise ValueError("Low-level cache checksum mismatch")
        return saved
    # Reuse only complete, checkpoint-identified DIP shards from the original run.
    old = existing_shards(root, digest, subject, settings.vgg)
    if all((old / f"{i:05d}.pt").is_file() for i in range(200)):
        images = []
        for i in tqdm(range(200), desc=f"sub-{subject:02d} cached DIP", unit="image"):
            saved = torch.load(old / f"{i:05d}.pt", map_location="cpu", weights_only=True)
            if saved["image_id"] != i or saved["checkpoint_sha256"] != digest:
                raise ValueError("Original DIP checkpoint/image mismatch")
            images.append(saved["recon"])
        low = torch.stack(images)
    else:
        predictions = output / "tokens"
        predictions.mkdir(parents=True, exist_ok=True)
        original = root / "evaluation" / digest / "predictions"
        if settings.vgg == "pooled":
            original = original / f"sub-{subject:02d}"
        for i in tqdm(range(200), desc=f"sub-{subject:02d} VGG prediction", unit="image"):
            dest = predictions / f"{i:05d}.pt"
            if dest.exists():
                continue
            if (original / dest.name).exists():
                saved = torch.load(original / dest.name, map_location="cpu", weights_only=True)
                if saved["image_id"] != i or saved["checkpoint_sha256"] != digest:
                    raise ValueError("Original VGG prediction mismatch")
                tokens = saved["tokens"]
            else:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    tokens = (
                        model.predict_all(torch.from_numpy(x[i : i + 1].copy()).cuda())[0]
                        .float()
                        .cpu()
                        .half()
                    )
            if tokens.shape != (7190, 512) or not torch.isfinite(tokens).all():
                raise ValueError("Invalid VGG prediction")
            torch_save(dest, tokens)
        # VGG tokenizer construction must not advance DIP weight-initialization RNG.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            tokenizer = BrainITVGGTokenizer().cuda().eval()
        del model
        gc.collect()
        torch.cuda.empty_cache()
        generator = torch.Generator(device="cuda").manual_seed(42)
        state_path = output / "dip_state.pt"
        state = (
            torch.load(state_path, map_location="cpu", weights_only=True)
            if state_path.exists()
            else None
        )
        low_batches = [] if state is None else state["batches"]
        start = 0 if state is None else state["completed"]
        if state is not None:
            torch.set_rng_state(state["torch_rng"])
            torch.cuda.set_rng_state(state["cuda_rng"])
            generator.set_state(state["generator_rng"])
        for start in tqdm(
            range(start, 200, 16), desc=f"sub-{subject:02d} DIP batches", unit="batch"
        ):
            tokens = torch.stack(
                [
                    torch.load(predictions / f"{i:05d}.pt", weights_only=True)
                    for i in range(start, min(start + 16, 200))
                ]
            ).float()
            low_batches.append(
                invert(tokens_to_brainit_vgg_features(tokens), tokenizer, generator).half()
            )
            torch_save(
                state_path,
                {
                    "completed": min(start + 16, 200),
                    "batches": low_batches,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(),
                    "generator_rng": generator.get_state(),
                },
            )
            commit()
        low = torch.cat(low_batches)
    if low.shape != (200, 3, 256, 256) or not torch.isfinite(low).all():
        raise ValueError("Invalid low-level reconstruction tensor")
    torch_save(output / "lowlevel.pt", low.half())
    report = {
        "subject": subject,
        "N": 200,
        "sha256": sha256(output / "lowlevel.pt"),
        "checkpoint_sha256": digest,
        "reused_original_shards": str(old) if old.exists() else None,
    }
    write_json(output / "complete.json", report)
    commit()
    return report
