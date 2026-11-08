"""One Modal entry point for VGG/DIP, Perceptogram CLIP, diffusion, and evaluation."""

import argparse
import csv
from pathlib import Path
import subprocess
import sys

import modal
from common import DATA, HERE, Settings, aggregate, guard, read_json, sha256, write_json

APP = "cortexdecoder-eeg-pipeline"
volume = modal.Volume.from_name("eeg-rpa-data")


def image(python, torch_version, vision_version, requirements):
    return (
        modal.Image.debian_slim(python_version=python)
        .pip_install(
            f"torch=={torch_version}",
            f"torchvision=={vision_version}",
            index_url="https://download.pytorch.org/whl/"
            + ("cu121" if torch_version == "2.4.0" else "cu129"),
        )
        .pip_install_from_requirements(str(HERE / requirements))
        .env(
            {
                "PYTHONUNBUFFERED": "1",
                "OMP_NUM_THREADS": "8",
                "TOKENIZERS_PARALLELISM": "false",
                "TORCH_HOME": str(DATA / "model_cache/torch"),
                "HF_HOME": str(DATA / "model_cache/huggingface"),
                "PYTHONWARNINGS": "ignore::FutureWarning",
            }
        )
        .workdir("/root/pipeline")
        .add_local_dir(HERE, "/root/pipeline", ignore=["results", ".runs", "__pycache__", "*.pyc"])
    )


vgg_image = image("3.11", "2.8.0", "0.23.0", "requirements-vgg.txt")
clip_image = image("3.10", "2.4.0", "0.19.0", "requirements-clip.txt")
eval_image = image("3.12", "2.8.0", "0.23.0", "requirements-eval.txt")
cpu_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("numpy==1.26.4", "scipy==1.13.1", "scikit-learn==1.6.1")
    .env({"OMP_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "8", "PYTHONUNBUFFERED": "1"})
    .workdir("/root/pipeline")
    .add_local_dir(HERE, "/root/pipeline", ignore=["results", ".runs", "__pycache__", "*.pyc"])
)
app = modal.App(APP)


def settings(values):
    return Settings(**{**values, "subjects": tuple(values["subjects"])})


def source_identity():
    return {
        p.name: sha256(p) for p in sorted(HERE.iterdir()) if p.suffix in (".py", ".json", ".txt")
    }


@app.function(
    image=vgg_image,
    gpu="H100",
    cpu=8,
    memory=65536,
    timeout=86400,
    volumes={str(DATA): volume},
    max_containers=1,
)
def vgg(values, subject):
    from vgg import run

    volume.reload()
    return run(settings(values), subject, volume.commit)


@app.function(
    image=clip_image,
    gpu="H100",
    cpu=8,
    memory=32768,
    timeout=21600,
    volumes={str(DATA): volume},
    max_containers=1,
)
def clip_targets():
    from clip import targets

    volume.reload()
    return targets(volume.commit)


@app.function(
    image=cpu_image,
    cpu=8,
    memory=16384,
    timeout=7200,
    volumes={str(DATA): volume},
    max_containers=1,
)
def clip(values, subject):
    from clip import run

    volume.reload()
    return run(settings(values), subject, volume.commit)


@app.function(
    image=clip_image,
    gpu="H100",
    cpu=8,
    memory=32768,
    timeout=14400,
    volumes={str(DATA): volume},
    max_containers=1,
)
def diffusion(values, subject):
    from diffusion import run

    volume.reload()
    return run(settings(values), subject, volume.commit)


@app.function(
    image=eval_image,
    gpu="H100",
    cpu=8,
    memory=65536,
    timeout=7200,
    volumes={str(DATA): volume},
    max_containers=1,
)
def evaluate(values, subject):
    from evaluate import run

    volume.reload()
    return run(settings(values), subject, volume.commit)


def save_summary(config, reports):
    report = aggregate(reports)
    report["requested_subjects"] = list(config.subjects)
    report["complete"] = len(reports) == len(config.subjects)
    write_json(config.output / "summary.json", report)
    rows = report["per_subject"]
    if rows:
        with (config.output / "per_subject.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        flattened = []
        for entry in report["aggregate"]:
            row = {k: v for k, v in entry.items() if not isinstance(v, dict)}
            for key, value in entry.items():
                if isinstance(value, dict):
                    row[key + "_mean"], row[key + "_std"] = value["mean"], value["std"]
            flattened.append(row)
        with (config.output / "aggregate.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(flattened[0]))
            writer.writeheader()
            writer.writerows(flattened)
    return report


@app.function(
    image=cpu_image,
    cpu=1,
    memory=4096,
    timeout=86400,
    volumes={str(DATA): volume},
    max_containers=1,
)
def workflow(values, stage):
    config = settings(values)
    volume.reload()
    guard(config.output / "run.json", {"settings": values, "source": source_identity()})
    stages = ["vgg", "clip", "diffusion", "evaluate"] if stage == "all" else [stage]
    reports = []
    try:
        if "clip" in stages:
            clip_targets.remote()
        for subject in config.subjects:
            for name in stages:
                volume.reload()
                write_json(
                    config.output / "progress.json",
                    {"subject": subject, "stage": name, "status": "running"},
                )
                volume.commit()
                result = {"vgg": vgg, "clip": clip, "diffusion": diffusion, "evaluate": evaluate}[
                    name
                ].remote(values, subject)
                if name == "evaluate":
                    reports.append(result)
                    volume.reload()
                    save_summary(config, reports)
                    volume.commit()
        volume.reload()
        report = (
            save_summary(config, reports) if reports else {"status": "complete", "stage": stage}
        )
        write_json(
            config.output / "progress.json",
            {"status": "complete", "stage": stage, "subjects": list(config.subjects)},
        )
        volume.commit()
        return report
    except Exception as exc:
        volume.reload()
        write_json(
            config.output / "progress.json", {"status": "failed", "stage": stage, "error": str(exc)}
        )
        volume.commit()
        raise


def subjects(value):
    try:
        result = tuple(range(1, 11)) if value == "all" else tuple(int(s) for s in value.split(","))
        if (
            not result
            or len(set(result)) != len(result)
            or any(s not in range(1, 11) for s in result)
        ):
            raise ValueError
        return result
    except ValueError:
        raise argparse.ArgumentTypeError("Use all or distinct subject numbers such as 1,3,7")


def download(config):
    destination = HERE / "results" / config.run
    remote = str(config.output.relative_to(DATA))
    for entry in volume.iterdir(remote, recursive=True):
        # Keep multi-GB VGG tokens and lowlevel tensors remote; download reviewable results.
        if not entry.path.endswith((".json", ".csv", ".png")):
            continue
        relative = Path(entry.path).relative_to(remote)
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            volume.read_file_into_fileobj(entry.path, stream)
    print("Downloaded:", destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["start", "resume", "status", "logs", "download"])
    parser.add_argument(
        "--run", required=True, help="Output name; old experiments are never modified"
    )
    parser.add_argument("--subjects", type=subjects, default=(1,), help="1, 1,3,7, or all")
    parser.add_argument("--vgg", choices=["auto", "subject01", "pooled"], default="auto")
    parser.add_argument(
        "--stage",
        choices=["all", "vgg", "clip", "diffusion", "evaluate"],
        help="Default all; resume keeps the previous stage",
    )
    parser.add_argument(
        "--images",
        type=int,
        default=200,
        help="First N official test images for diffusion/evaluation",
    )
    parser.add_argument("--sampler", choices=["perceptogram", "img2img"], default="perceptogram")
    parser.add_argument("--strength", type=float, default=0.4, help="Used only by img2img")
    parser.add_argument(
        "--guidance", type=float, help="Default 7.5 for Perceptogram, 5 for img2img"
    )
    args = parser.parse_args()
    record = HERE / ".runs" / (args.run + ".json")
    if args.action == "start":
        if record.exists():
            parser.error("Run record exists; use resume or choose a new run name")
        profile = (
            ("subject01" if args.subjects == (1,) else "pooled") if args.vgg == "auto" else args.vgg
        )
        guidance = (
            args.guidance
            if args.guidance is not None
            else (7.5 if args.sampler == "perceptogram" else 5.0)
        )
        config = Settings(
            args.run, args.subjects, profile, args.images, args.sampler, args.strength, guidance
        )
        saved = {"settings": config.to_dict(), "stage": args.stage or "all"}
    else:
        saved = read_json(record)
        config = settings(saved["settings"])
        if args.action == "resume" and args.stage is not None:
            saved["stage"] = args.stage
    if args.action in ("start", "resume"):
        if args.action == "resume":
            try:
                modal.FunctionCall.from_id(saved["call_id"]).get(timeout=0)
            except TimeoutError:
                parser.error("This run is still active; use logs or status")
            except Exception:
                pass  # Failed/expired calls resume from committed stage artifacts.
        subprocess.run(
            [sys.executable, "-m", "modal", "deploy", str(Path(__file__).resolve())], check=True
        )
        call = modal.Function.from_name(APP, "workflow").spawn(saved["settings"], saved["stage"])
        saved["call_id"] = call.object_id
        write_json(record, saved)
        print("Started:", call.object_id, "\nSettings:", saved["settings"])
    elif args.action == "logs":
        subprocess.run([sys.executable, "-m", "modal", "app", "logs", APP, "-f"], check=True)
    elif args.action == "download":
        download(config)
    else:
        remote = str(config.output.relative_to(DATA))
        for name in ("progress.json", "summary.json"):
            try:
                print(b"".join(volume.read_file(remote + "/" + name)).decode())
            except FileNotFoundError:
                print(name + ": not yet saved")


if __name__ == "__main__":
    main()
