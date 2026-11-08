"""Shared paths, settings, and small artifact helpers."""

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import os

HERE = Path(__file__).resolve().parent
DATA = Path("/mnt/eeg-rpa")
EEG_CLIP = DATA / "perceptogram_vd_release_54db0981/data"
OLD_TARGETS = DATA / "perceptogram_linear_retrieval_54db0981_v1/targets"
MODEL = json.loads((HERE / "model_assets.json").read_text())
VGG_RUNS = {
    "subject01": (
        "eeg_bridge/sub01_temporal_vgg_contrastive_seed42_01",
        "c1af9727dcb88da88063c7381356be346985fea9bef2ecb21edfa5b276003617",
    ),
    "pooled": (
        "eeg_bridge_transfer_learn/all10_temporal_vgg_seed42_01/all_subjects",
        "53ebf4170a2a9f132a4f1fa56126dcaec2a7ead823b4723dd36795d75684505d",
    ),
}


@dataclass(frozen=True)
class Settings:
    run: str
    subjects: tuple = (1,)
    vgg: str = "subject01"
    images: int = 200
    sampler: str = "perceptogram"
    strength: float = 0.4
    guidance: float = 7.5
    steps: int = 20
    seed: int = 0

    def __post_init__(self):
        if not self.run or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for c in self.run
        ):
            raise ValueError("Use letters, digits, underscores, or hyphens for the run name")
        if (
            not self.subjects
            or len(set(self.subjects)) != len(self.subjects)
            or any(s not in range(1, 11) for s in self.subjects)
        ):
            raise ValueError("Choose distinct subjects from 1 through 10")
        if self.vgg not in VGG_RUNS or (self.vgg == "subject01" and self.subjects != (1,)):
            raise ValueError(
                "The subject01 checkpoint supports subject 1 only; use pooled for other subjects"
            )
        if not 2 <= self.images <= 200 or self.sampler not in ("perceptogram", "img2img"):
            raise ValueError("Choose 2..200 images and perceptogram or img2img sampling")
        if (
            not 0 < self.strength <= 1
            or self.steps < 1
            or int(self.steps * self.strength) < 1
            or self.guidance <= 1
        ):
            raise ValueError("Invalid diffusion settings; classifier-free guidance must exceed 1")

    @property
    def output(self):
        return DATA / "eeg_pipeline" / self.run

    def subject_dir(self, subject):
        return self.output / f"sub-{subject:02d}"

    def to_dict(self):
        return asdict(self)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def torch_save(path, value):
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def guard(path, identity):
    """Reject incompatible caches rather than silently mixing runs."""
    identity = json.loads(json.dumps(identity))
    if Path(path).exists() and read_json(path) != identity:
        raise ValueError(f"Existing recipe differs: {path}; choose a new run name")
    write_json(path, identity)


def check_order():
    import numpy as np

    manifest = read_json(DATA / "images.json")
    metadata = np.load(
        EEG_CLIP / "thingseeg2_metadata/image_metadata.npy", allow_pickle=True
    ).item()
    for split, prefix in [("training_images", "train"), ("test_images", "test")]:
        expected = [
            f"{c}/{f}"
            for c, f in zip(metadata[prefix + "_img_concepts"], metadata[prefix + "_img_files"])
        ]
        actual = ["/".join(p.split("/")[-2:]) for p in manifest[split]["images"]]
        if actual != expected:
            raise ValueError(f"VGG and CLIP stimulus order differs: {split}")


def aggregate(reports):
    """Equal subject weighting; never average EEG or image predictions across people."""
    import numpy as np

    rows = [row for report in reports for row in report]
    result = []
    for method in dict.fromkeys(row["Method"] for row in rows):
        group = [r for r in rows if r["Method"] == method]
        entry = {"Method": method, "subjects": len(group), "N_per_subject": group[0]["N"]}
        for key in group[0]:
            if key in ("Subject", "Method", "N"):
                continue
            values = [r[key] for r in group]
            entry[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            }
        result.append(entry)
    return {
        "per_subject": rows,
        "aggregate": result,
        "weighting": "arithmetic mean across selected subjects; sample SD",
    }
