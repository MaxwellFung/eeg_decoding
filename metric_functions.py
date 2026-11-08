"""Production-notebook metric functions, retained verbatim for score parity."""

import numpy as np
import torch
from torch.nn import functional as F
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim_fn
from tqdm.auto import tqdm

DEVICE = torch.device("cuda")
FINAL_EVAL_FEATURE_BATCH_SIZE = 32
SEED = 42
# Models are loaded lazily by evaluate.load_models().
alex = inception = effnet = clip_eval = swav = None
imagenet_mean, imagenet_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
clip_mean = [0.48145466, 0.4578275, 0.40821073]
clip_std = [0.26862954, 0.26130258, 0.27577711]


def norm_batch(x, size, mean, std):
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    mean = torch.tensor(mean, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


@torch.no_grad()
def extract_features(images, fn, batch_size=FINAL_EVAL_FEATURE_BATCH_SIZE):
    outs = []
    for start in tqdm(range(0, len(images), batch_size), desc="feature batches", leave=False):
        outs.append(fn(images[start : start + batch_size].to(DEVICE)).float().flatten(1).cpu())
    return torch.cat(outs, dim=0)


def two_way_score(recon_feats, true_feats):
    r = np.corrcoef(true_feats.numpy(), recon_feats.numpy())[: len(true_feats), len(true_feats) :]
    congruent = np.diag(r)
    return float(np.mean(np.sum(r < congruent, axis=0)) / (len(true_feats) - 1))


def correlation_distance(recon_feats, true_feats):
    a, b = recon_feats.numpy(), true_feats.numpy()
    return float(np.mean([1.0 - np.corrcoef(a[i], b[i])[0, 1] for i in range(len(a))]))


def retrieval_score(pred_feats, true_feats, n_loops=30, n_samples=300, seed=SEED):
    pred = F.normalize(pred_feats.float(), dim=-1)
    true = F.normalize(true_feats.float(), dim=-1)
    n = len(pred)
    fwd, bwd = [], []
    rng = np.random.RandomState(seed)
    for _ in range(n_loops):
        idx = torch.as_tensor(
            rng.choice(n, size=min(n_samples, n), replace=False), dtype=torch.long
        )
        labels = torch.arange(len(idx))
        fwd.append((pred[idx] @ true[idx].T).argmax(1).eq(labels).float().mean().item())
        bwd.append((true[idx] @ pred[idx].T).argmax(1).eq(labels).float().mean().item())
    return float(np.mean(fwd)), float(np.mean(bwd))


def pixel_metrics(recons, images):
    resized_recons = F.interpolate(recons, size=(425, 425), mode="bilinear", align_corners=False)
    resized_images = F.interpolate(images, size=(425, 425), mode="bilinear", align_corners=False)
    pixcorr = np.mean(
        [
            np.corrcoef(a.flatten(), b.flatten())[0, 1]
            for a, b in zip(resized_recons.numpy(), resized_images.numpy())
        ]
    )
    rec_gray = rgb2gray(resized_recons.permute(0, 2, 3, 1).numpy())
    img_gray = rgb2gray(resized_images.permute(0, 2, 3, 1).numpy())
    ssim = np.mean(
        [
            ssim_fn(
                r, i, data_range=1.0, gaussian_weights=True, sigma=1.5, use_sample_covariance=False
            )
            for r, i in zip(rec_gray, img_gray)
        ]
    )
    return float(pixcorr), float(ssim)


def alex_feats(x, key):
    return extract_features(x, lambda b: alex(norm_batch(b, 256, imagenet_mean, imagenet_std))[key])


def inception_feats(x):
    return extract_features(
        x, lambda b: inception(norm_batch(b, 342, imagenet_mean, imagenet_std))["pool"]
    )


def eff_feats(x):
    return extract_features(
        x, lambda b: effnet(norm_batch(b, 255, imagenet_mean, imagenet_std))["pool"]
    )


def clip_feats(x):
    return extract_features(
        x, lambda b: clip_eval.encode_image(norm_batch(b, 224, clip_mean, clip_std))
    )


def swav_feats(x):
    return extract_features(
        x, lambda b: swav(norm_batch(b, 224, imagenet_mean, imagenet_std))["pool"]
    )
