"""Perceptogram unCLIP sampling, optionally using scheduler-consistent img2img."""

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from common import MODEL, guard, read_json, sha256, write_json


def load_pipeline():
    from diffusers import StableUnCLIPImg2ImgPipeline

    pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
        MODEL["repository"], revision=MODEL["revision"], torch_dtype=torch.float16, variant="fp16"
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


@torch.no_grad()
def sample(pipe, low, prediction, settings):
    """The released decode loop, restricted to one image and an empty text prompt."""
    rgb = (low.float().clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    rgb = np.array(Image.fromarray(rgb).resize((768, 768), Image.Resampling.LANCZOS)).copy()
    tensor = torch.from_numpy(rgb).permute(2, 0, 1)[None].cuda().half()
    tensor = tensor / 255 if settings.sampler == "perceptogram" else tensor / 127.5 - 1
    latent = pipe.vae.encode(tensor, return_dict=False)[0].mode()
    torch.manual_seed(settings.seed)
    noise = torch.randn(latent.shape, device="cuda").half()
    pipe.scheduler.set_timesteps(settings.steps, device="cuda")
    timesteps = pipe.scheduler.timesteps
    if settings.sampler == "perceptogram":
        latent = latent * 0.02 + noise
    else:
        start = settings.steps - min(int(settings.steps * settings.strength), settings.steps)
        timesteps = timesteps[start * pipe.scheduler.order :]
        if hasattr(pipe.scheduler, "set_begin_index"):
            pipe.scheduler.set_begin_index(start * pipe.scheduler.order)
        latent = pipe.scheduler.add_noise(
            latent * pipe.vae.config.scaling_factor, noise, timesteps[:1]
        )
    if not len(timesteps) or not torch.isfinite(latent).all():
        raise ValueError("Invalid diffusion initialization")
    embedding = torch.from_numpy(prediction.copy()).cuda().half()
    extra = torch.cat(
        [
            torch.ones(512, device="cuda", dtype=torch.float16),
            torch.zeros(512, device="cuda", dtype=torch.float16),
        ]
    )
    embedding = torch.cat([embedding, extra])[None]
    conditioning = torch.cat([torch.zeros_like(embedding), embedding])
    torch.manual_seed(settings.seed)
    positive, negative = pipe.encode_prompt("", "cuda", 1, True)
    prompt = torch.cat([negative, positive])
    extra_kwargs = pipe.prepare_extra_step_kwargs(None, 0.0)
    for timestep in timesteps:
        inputs = pipe.scheduler.scale_model_input(torch.cat([latent] * 2), timestep)
        predicted = pipe.unet(
            inputs,
            timestep,
            encoder_hidden_states=prompt,
            class_labels=conditioning,
            cross_attention_kwargs=None,
            return_dict=False,
        )[0]
        unconditional, conditional = predicted.chunk(2)
        guided = unconditional + settings.guidance * (conditional - unconditional)
        latent = pipe.scheduler.step(guided, timestep, latent, **extra_kwargs, return_dict=False)[0]
    decoded = pipe.vae.decode(latent / pipe.vae.config.scaling_factor, return_dict=False)[0]
    return pipe.image_processor.postprocess(decoded, output_type="pil")[0]


def run(settings, subject, commit=lambda: None):
    root = settings.subject_dir(subject)
    low_path, clip_path = root / "vgg/lowlevel.pt", root / "clip/predictions.npy"
    for stage, path in [("vgg", low_path), ("clip", clip_path)]:
        if sha256(path) != read_json(root / stage / "complete.json")["sha256"]:
            raise ValueError(f"{stage} input checksum mismatch")
    output = root / "diffusion"
    identity = {
        "lowlevel_sha256": sha256(low_path),
        "clip_sha256": sha256(clip_path),
        "sampler": settings.sampler,
        "strength": settings.strength if settings.sampler == "img2img" else None,
        "steps": settings.steps,
        "guidance": settings.guidance,
        "seed_per_image": settings.seed,
        "images": settings.images,
        "model_revision": MODEL["revision"],
        "implementation_sha256": sha256(__file__),
    }
    guard(output / "recipe.json", identity)
    if (output / "complete.json").exists():
        report = read_json(output / "complete.json")
        for name, digest in report["sha256"].items():
            if sha256(output / name) != digest:
                raise ValueError("Final image checksum mismatch")
        return report
    low = torch.load(low_path, map_location="cpu", weights_only=True).float()
    predictions = np.load(clip_path)
    if low.shape != (200, 3, 256, 256) or predictions.shape != (200, 1024):
        raise ValueError("Expected 200 aligned DIP images and CLIP predictions")
    pipe = load_pipeline()
    for i in tqdm(range(settings.images), desc=f"sub-{subject:02d} unCLIP", unit="image"):
        path = output / f"{i:05d}.png"
        if path.exists():
            with Image.open(path) as image:
                image.verify()
            continue
        result = sample(pipe, low[i], predictions[i], settings)
        temporary = path.with_name("." + path.name)
        result.save(temporary)
        temporary.replace(path)
        write_json(output / "progress.json", {"completed": i + 1, "total": settings.images})
        commit()
    report = {
        "subject": subject,
        "N": settings.images,
        "sha256": {f"{i:05d}.png": sha256(output / f"{i:05d}.png") for i in range(settings.images)},
    }
    write_json(output / "complete.json", report)
    commit()
    return report
