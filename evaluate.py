"""The same low-level/final evaluators and previews as the completed subject-01 run."""

import csv
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image, ImageDraw

from common import DATA, guard, read_json, sha256, write_json
import metric_functions as metric


def load_models():
    import open_clip
    from torchvision.models import (
        alexnet,
        AlexNet_Weights,
        efficientnet_b1,
        EfficientNet_B1_Weights,
        inception_v3,
        Inception_V3_Weights,
    )
    from torchvision.models.feature_extraction import create_feature_extractor

    metric.alex = (
        create_feature_extractor(
            alexnet(weights=AlexNet_Weights.IMAGENET1K_V1),
            return_nodes={"features.4": "a2", "features.11": "a5"},
        )
        .cuda()
        .eval()
        .requires_grad_(False)
    )
    metric.inception = (
        create_feature_extractor(
            inception_v3(weights=Inception_V3_Weights.DEFAULT), return_nodes={"avgpool": "pool"}
        )
        .cuda()
        .eval()
        .requires_grad_(False)
    )
    metric.effnet = (
        create_feature_extractor(
            efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT),
            return_nodes={"avgpool": "pool"},
        )
        .cuda()
        .eval()
        .requires_grad_(False)
    )
    metric.clip_eval, _, _ = open_clip.create_model_and_transforms(
        "ViT-L-14",
        pretrained="openai",
        device="cuda",
        cache_dir=str(DATA / "model_cache/open_clip"),
    )
    metric.clip_eval = metric.clip_eval.eval().requires_grad_(False)
    metric.swav = (
        create_feature_extractor(
            torch.hub.load("facebookresearch/swav:main", "resnet50"),
            return_nodes={"avgpool": "pool"},
        )
        .cuda()
        .eval()
        .requires_grad_(False)
    )


def ground_truth(count):
    """Match StimulusImages: RGB floats, bicubic 256px, no PIL resize/center crop."""
    manifest = read_json(DATA / "images.json")
    images = []
    for relative in manifest["test_images"]["images"][:count]:
        with Image.open(Path(manifest["root"]) / relative) as image:
            tensor = (
                torch.from_numpy(np.array(image.convert("RGB")).copy()).permute(2, 0, 1).float()
                / 255
            )
            images.append(
                F.interpolate(tensor[None], (256, 256), mode="bicubic", align_corners=False)[
                    0
                ].clamp(0, 1)
            )
    return torch.stack(images)


def previews(output, truth, low, final, subject):
    for start in range(0, len(truth), 20):
        stop = min(start + 20, len(truth))
        canvas = Image.new("RGB", (1552, ((stop - start + 1) // 2) * 284 + 44), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (10, 10),
            f"Subject {subject:02d}: ground truth | VGG DIP | final prediction",
            fill="black",
        )
        for k, i in enumerate(range(start, stop)):
            x, y = (k % 2) * 784, 44 + (k // 2) * 284
            draw.text((x + 4, y), f"Test image {i:03d}: actual | low-level | final", fill="black")
            for j, batch in enumerate([truth, low, final]):
                rgb = batch[i].clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).numpy()
                canvas.paste(Image.fromarray(rgb), (x + j * 256, y + 24))
        canvas.save(output / f"preview_{start:03d}.png")


def run(settings, subject, commit=lambda: None):
    root = settings.subject_dir(subject)
    output = root / "evaluation"
    low_path = root / "vgg/lowlevel.pt"
    done = read_json(root / "diffusion/complete.json")
    if done["N"] != settings.images:
        raise ValueError("Final image count differs")
    if sha256(low_path) != read_json(root / "vgg/complete.json")["sha256"]:
        raise ValueError("Low-level checksum mismatch")
    identity = {
        "N": settings.images,
        "lowlevel_sha256": sha256(low_path),
        "final_sha256": done["sha256"],
        "ground_truth_manifest_sha256": sha256(DATA / "images.json"),
        "implementation": {
            n: sha256(Path(__file__).with_name(n)) for n in ("evaluate.py", "metric_functions.py")
        },
    }
    guard(output / "recipe.json", identity)
    if (output / "metrics.json").exists():
        return read_json(output / "metrics.json")
    low = (
        torch.load(low_path, map_location="cpu", weights_only=True)[: settings.images]
        .float()
        .clamp(0, 1)
    )
    final = []
    for i in range(settings.images):
        path = root / "diffusion" / f"{i:05d}.png"
        if sha256(path) != done["sha256"][path.name]:
            raise ValueError("Final image checksum mismatch")
        with Image.open(path) as image:
            final.append(
                torch.from_numpy(np.array(image.convert("RGB")).copy()).permute(2, 0, 1).float()
                / 255
            )
    final = F.interpolate(
        torch.stack(final), (256, 256), mode="bicubic", align_corners=False
    ).clamp(0, 1)
    truth = ground_truth(settings.images)
    load_models()
    extractors = {
        "AlexNet(2)": lambda x: metric.alex_feats(x, "a2"),
        "AlexNet(5)": lambda x: metric.alex_feats(x, "a5"),
        "InceptionV3": metric.inception_feats,
        "CLIP": metric.clip_feats,
        "EffNet-B": metric.eff_feats,
        "SwAV": metric.swav_feats,
    }
    real = {key: fn(truth) for key, fn in extractors.items()}
    rows = []
    for label, images in [("Low-Level", low), ("Final", final)]:
        pix, ssim = metric.pixel_metrics(images.cpu(), truth.cpu())
        row = {
            "Subject": f"sub-{subject:02d}",
            "Method": label,
            "N": settings.images,
            "PixCorr": pix,
            "SSIM": ssim,
        }
        for key, fn in extractors.items():
            predicted = fn(images)
            row[key] = (
                metric.correlation_distance if key in ("EffNet-B", "SwAV") else metric.two_way_score
            )(predicted, real[key])
            if key == "CLIP":
                row["ImageRetrieval"], row["BrainRetrieval"] = metric.retrieval_score(
                    predicted, real[key], n_samples=settings.images, seed=42
                )
        rows.append(row)
    previews(output, truth, low, final, subject)
    with (output / "metrics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Completion marker is written after every metric and preview has succeeded.
    write_json(output / "metrics.json", rows)
    commit()
    print(f"sub-{subject:02d} evaluation: {rows}", flush=True)
    return rows
