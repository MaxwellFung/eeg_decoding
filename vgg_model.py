"""Checkpoint-compatible EEG bridge, VGG features, and grouped DIP network."""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
from torchvision import models

TOKEN_COUNTS = (3136, 3025, 784, 196, 49)
VGG_LAYER_NAMES = ("relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3")
VGG_IMAGE_SIZE = 112
DIP_INPUT_DIM = 32
DIP_INTERNAL_DIM = 128


class CrossAttentionVGGReadout(nn.Module):
    """Learned VGG-position queries decode all supplied brain tokens.

    This matches the original CrossAttentionBridge dimensions and operations:
    learned query table, optional brain projection, memory LayerNorm, four
    TransformerDecoder layers, and the final feature projection.
    """

    def __init__(
        self,
        brain_dim,
        target_dim,
        num_queries,
        d_model=768,
        n_layers=4,
        n_heads=8,
        ffn_dim=3072,
        dropout=0.1,
    ):
        super().__init__()
        self.query_embed = nn.Parameter(torch.randn(num_queries, d_model) * 0.02)
        self.brain_proj = nn.Linear(brain_dim, d_model) if brain_dim != d_model else nn.Identity()
        self.brain_norm = nn.LayerNorm(d_model)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.output_proj = nn.Linear(d_model, target_dim)


class EEGVGG(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.encoder = nn.Conv1d(17, c.patch_dim, c.patch_size, stride=c.patch_stride)
        self.patch_norm = nn.LayerNorm(c.patch_dim)
        self.position = nn.Parameter(
            torch.randn(1, 1 + (175 - c.patch_size) // c.patch_stride, c.patch_dim) * 0.02
        )
        self.dropout = nn.Dropout(c.dropout)
        self.head = CrossAttentionVGGReadout(c.patch_dim, 512, sum(TOKEN_COUNTS), dropout=c.dropout)

    def encode(self, x):
        if x.ndim != 3 or x.shape[1:] != (17, 175):
            raise ValueError("Expected [B,17,175] EEG")
        return self.dropout(
            torch.nn.functional.gelu(self.patch_norm(self.encoder(x).transpose(1, 2)))
            + self.position
        )

    def decode(self, memory, indices):
        h = self.head
        m = h.brain_norm(h.brain_proj(memory))
        q = h.query_embed[indices][None].expand(len(memory), -1, -1)
        for layer in h.decoder.layers:
            q = (
                checkpoint(layer, q, m, use_reentrant=False)
                if self.c.gradient_checkpointing and self.training
                else layer(q, m)
            )
        if h.decoder.norm is not None:
            q = h.decoder.norm(q)
        return h.output_proj(q)

    def predict_all(self, x):
        memory = self.encode(x)
        return torch.cat(
            [
                self.decode(
                    memory,
                    torch.arange(
                        s, min(s + self.c.prediction_chunk, sum(TOKEN_COUNTS)), device=x.device
                    ),
                )
                for s in range(0, sum(TOKEN_COUNTS), self.c.prediction_chunk)
            ],
            dim=1,
        )


def split_all_vgg_tokens(tokens):
    return list(tokens.split(TOKEN_COUNTS, dim=1))


class BrainITVGGTokenizer(nn.Module):
    """VGG16-BN feature maps and tokenization described in BrainIT Appendix D.2."""

    def __init__(self):
        super().__init__()
        weights = models.VGG16_BN_Weights.DEFAULT
        self.features = models.vgg16_bn(weights=weights).features.eval()
        self.capture = {5: 0, 12: 1, 22: 2, 32: 3, 42: 4}
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        for p in self.parameters():
            p.requires_grad_(False)

    def extract_features(self, images01):
        x = F.interpolate(
            images01, size=(VGG_IMAGE_SIZE, VGG_IMAGE_SIZE), mode="bicubic", align_corners=False
        ).clamp(0, 1)
        x = (x - self.mean) / self.std
        outs = [None] * len(VGG_LAYER_NAMES)
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.capture:
                outs[self.capture[i]] = x
        return outs


def reduce_repeated_token_dim(tokens, true_dim):
    if tokens.shape[-1] == true_dim:
        return tokens
    if tokens.shape[-1] % true_dim != 0:
        return tokens[..., :true_dim]
    return tokens.reshape(*tokens.shape[:-1], tokens.shape[-1] // true_dim, true_dim).mean(dim=-2)


def fold_overlapping_tokens(tokens, channels, output_hw, kernel_size, stride):
    cols = tokens.transpose(1, 2).contiguous()
    feat = F.fold(cols, output_size=output_hw, kernel_size=kernel_size, stride=stride)
    ones = torch.ones(
        (1, channels * kernel_size * kernel_size, tokens.shape[1]),
        device=tokens.device,
        dtype=tokens.dtype,
    )
    counts = F.fold(ones, output_size=output_hw, kernel_size=kernel_size, stride=stride).clamp_min(
        1e-06
    )
    return feat / counts


def tokens_to_brainit_vgg_features(tokens):
    layers = split_all_vgg_tokens(tokens.float())
    feat1_tokens = reduce_repeated_token_dim(layers[0], 4 * 64)
    feat2_tokens = reduce_repeated_token_dim(layers[1], 4 * 128)
    feat3_tokens = reduce_repeated_token_dim(layers[2], 256)
    feat4_tokens = reduce_repeated_token_dim(layers[3], 512)
    feat5_tokens = reduce_repeated_token_dim(layers[4], 512)
    feat1 = fold_overlapping_tokens(
        feat1_tokens, channels=64, output_hw=(112, 112), kernel_size=2, stride=2
    )
    feat2 = fold_overlapping_tokens(
        feat2_tokens, channels=128, output_hw=(56, 56), kernel_size=2, stride=1
    )
    feat3 = feat3_tokens.transpose(1, 2).reshape(tokens.shape[0], 256, 28, 28).contiguous()
    feat4 = feat4_tokens.transpose(1, 2).reshape(tokens.shape[0], 512, 14, 14).contiguous()
    feat5 = feat5_tokens.transpose(1, 2).reshape(tokens.shape[0], 512, 7, 7).contiguous()
    return [feat1, feat2, feat3, feat4, feat5]


class GroupedConvBlock(nn.Module):
    def __init__(self, groups, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(groups * in_ch, groups * out_ch, 3, padding=1, groups=groups),
            nn.BatchNorm2d(groups * out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(groups * out_ch, groups * out_ch, 3, padding=1, groups=groups),
            nn.BatchNorm2d(groups * out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class BatchedDIPUNet(nn.Module):
    def __init__(self, groups, input_dim=DIP_INPUT_DIM, width=DIP_INTERNAL_DIM):
        super().__init__()
        self.groups = int(groups)
        self.input_dim = int(input_dim)
        self.width = int(width)
        self.down1 = GroupedConvBlock(groups, input_dim, width)
        self.down2 = GroupedConvBlock(groups, width, width)
        self.down3 = GroupedConvBlock(groups, width, width)
        self.mid = GroupedConvBlock(groups, width, width)
        self.up3 = GroupedConvBlock(groups, width * 2, width)
        self.up2 = GroupedConvBlock(groups, width * 2, width)
        self.up1 = GroupedConvBlock(groups, width * 2, width)
        self.out = nn.Conv2d(groups * width, groups * 3, 1, groups=groups)

    def _merge(self, x):
        b, c, h, w = x.shape
        if b != self.groups:
            raise ValueError(f"expected batch={self.groups}, got {b}")
        return x.reshape(1, b * c, h, w)

    def _split(self, x, channels):
        _, _, h, w = x.shape
        return x.reshape(self.groups, channels, h, w)

    def forward(self, z):
        z = self._merge(z)
        e1 = self.down1(z)
        e2 = self.down2(F.avg_pool2d(e1, 2))
        e3 = self.down3(F.avg_pool2d(e2, 2))
        x = self.mid(F.avg_pool2d(e3, 2))
        x = F.interpolate(x, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up3(torch.cat([x, e3], dim=1))
        x = F.interpolate(x, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up2(torch.cat([x, e2], dim=1))
        x = F.interpolate(x, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        return self._split(torch.sigmoid(self.out(x)), 3)
