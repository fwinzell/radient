"""Feature encoders and Hugging Face loaders for the nnFoundation models."""

from __future__ import annotations

import torch
from torch import nn


CNN_REPO_ID = "MIC-DKFZ/nnFoundationCNN"
VIT_REPO_ID = "MIC-DKFZ/nnFoundationViT"
STAGE_CHANNELS = [32, 64, 128, 256, 320, 320]
VIT_INPUT_SHAPE = (192, 192, 192)
VIT_PATCH_EMBED_SIZE = (8, 8, 8)


def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _preprocess_nnfoundation_volume(volume: torch.Tensor) -> dict[str, torch.Tensor]:
    """Apply nnFoundation's volume normalization and make a 192-voxel patch.

    nnssl's training pipeline crops away the all-zero exterior, z-scores the
    cropped volume per case, then samples fixed-size training patches. At
    extraction time we use a deterministic center crop (or zero padding) in
    place of the random training crop. No voxel-spacing resampling is done.
    ``volume`` is the [C,D,H,W] tensor emitted for one sample by
    ``RadiologyProcessor``; a [B,C,D,H,W] tensor is also accepted.
    """
    if not isinstance(volume, torch.Tensor):
        volume = torch.as_tensor(volume)
    if volume.ndim == 4:
        volume = volume.unsqueeze(0)
    if volume.ndim != 5:
        raise ValueError(
            "nnFoundation preprocessing expects [C,D,H,W] or [B,C,D,H,W], "
            f"got shape {tuple(volume.shape)}"
        )
    if volume.shape[1] != 1:
        raise ValueError(f"nnFoundation expects one input channel, got {volume.shape[1]}")

    processed = []
    for sample in volume:
        # nnU-Net's preprocessing removes the all-zero exterior before
        # normalization. Find the bounding box over the single image channel.
        image = sample[0].to(dtype=torch.float32)
        nonzero = torch.nonzero(image != 0, as_tuple=False)
        if nonzero.numel():
            lower = nonzero.amin(dim=0)
            upper = nonzero.amax(dim=0) + 1
            image = image[
                lower[0] : upper[0],
                lower[1] : upper[1],
                lower[2] : upper[2],
            ]

        mean = image.mean()
        std = image.std(unbiased=False)
        image = (image - mean) / std.clamp_min(1e-8)

        # Center crop each axis; pad undersized axes symmetrically with zeros.
        crop_slices = []
        axis_padding = []
        for size, target in zip(image.shape, VIT_INPUT_SHAPE):
            if size > target:
                start = (size - target) // 2
                crop_slices.append(slice(start, start + target))
                axis_padding.append((0, 0))
            else:
                crop_slices.append(slice(None))
                total = target - size
                before = total // 2
                axis_padding.append((before, total - before))
        image = image[tuple(crop_slices)]
        # F.pad specifies dimensions from last to first (W, H, D).
        pad = [n for pair in reversed(axis_padding) for n in pair]
        if any(pad):
            # torch.nn.functional.pad lists dimensions in reverse order.
            image = torch.nn.functional.pad(image, pad)
        processed.append(image.unsqueeze(0))

    return {"pixel_values": torch.stack(processed, dim=0)}


def preprocess_nnfoundation_ct(volume: torch.Tensor) -> dict[str, torch.Tensor]:
    """Preprocess a CT volume as nnFoundation did: crop, per-case z-score.

    CT and MRI use the same normalization in the released nnFoundation
    checkpoints; CT values are not additionally windowed or HU-clipped.
    """
    return _preprocess_nnfoundation_volume(volume)


def preprocess_nnfoundation_mri(volume: torch.Tensor) -> dict[str, torch.Tensor]:
    """Preprocess an MRI volume as nnFoundation did: crop, per-case z-score."""
    return _preprocess_nnfoundation_volume(volume)


def _select_preprocessor(config: dict):
    modality = str(config.get("modality", "mri")).lower()
    if modality == "ct":
        return preprocess_nnfoundation_ct
    if modality in {"mri", "mr"}:
        return preprocess_nnfoundation_mri
    raise ValueError("nnFoundation modality must be 'ct' or 'mri'")


class nnFoundationCNN(nn.Module):
    """ResEnc-L encoder that returns its final feature embedding map.

    The nnFoundation checkpoint is loaded into the complete ResEnc-L network,
    but this module keeps only its encoder so the segmentation decoder and its
    upsampled predictions are not part of the forward path.
    """

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, x: torch.Tensor):
        return self.encoder(x)[-1]


class nnFoundationViT(nn.Module):
    """PrimusX encoder that returns transformer patch-token embeddings.

    This follows the nnFoundation ViT loading example: run the Primus patch
    projection and EVA transformer directly, bypassing Primus' up-projection
    and segmentation decoder.
    """

    def __init__(self, down_projection: nn.Module, eva: nn.Module):
        super().__init__()
        self.down_projection = down_projection
        self.eva = eva

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Primus projection outputs (B, C, Z', Y', X'); flatten the spatial
        # token grid while moving the embedding dimension to the last axis.
        x = self.down_projection(x).permute(0, 2, 3, 4, 1)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        embeddings = self.eva(x)
        # Eva returns its requested feature collection; expose the final
        # tensor-like embedding result as the model output.
        if isinstance(embeddings, (tuple, list)):
            embeddings = embeddings[0]
        return embeddings


def load_nnfoundation_cnn(config=None, device=None):
    """Load nnFoundationCNN and return ``(model, preprocessor, '3d')``."""
    from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
    from huggingface_hub import hf_hub_download

    network = ResidualEncoderUNet(
        input_channels=1,
        n_stages=6,
        features_per_stage=STAGE_CHANNELS,
        conv_op=nn.Conv3d,
        kernel_sizes=[[3, 3, 3]] * 6,
        strides=[[1, 1, 1]] + [[2, 2, 2]] * 5,
        n_blocks_per_stage=[1, 3, 4, 6, 6, 6],
        num_classes=1,
        n_conv_per_stage_decoder=[1] * 5,
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        deep_supervision=False,
    )
    checkpoint_path = hf_hub_download(CNN_REPO_ID, "checkpoint_final.pth")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    network.load_state_dict(checkpoint["network_weights"], strict=True)

    encoder = network.encoder
    encoder.return_skips = True
    model = nnFoundationCNN(encoder).to(device or _default_device()).eval()
    return model, _select_preprocessor(config or {}), "3d"


def load_nnfoundation_vit(config=None, device=None):
    """Load nnFoundationViT and return ``(model, preprocessor, '3d')``."""
    from dynamic_network_architectures.architectures.primus import Primus
    from huggingface_hub import hf_hub_download

    network = Primus(
        input_channels=1,
        embed_dim=1056,
        num_classes=1,
        eva_depth=40,
        eva_numheads=16,
        input_shape=VIT_INPUT_SHAPE,
        patch_embed_size=VIT_PATCH_EMBED_SIZE,
        init_values=0.1,
        scale_attn_inner=True,
    )
    checkpoint_path = hf_hub_download(VIT_REPO_ID, "checkpoint_final.pth")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    network.load_state_dict(checkpoint["network_weights"], strict=False)

    model = nnFoundationViT(network.down_projection, network.eva)
    model = model.to(device or _default_device()).eval()
    return model, _select_preprocessor(config or {}), "3d"


# Registry adapter retained for the common model-zoo interface. All loader
# arguments are optional for direct use; the registry supplies config/device.
def load_nnfoundation(config=None, device=None):
    return load_nnfoundation_cnn(config, device)


__all__ = [
    "nnFoundationCNN",
    "nnFoundationViT",
    "load_nnfoundation_cnn",
    "load_nnfoundation_vit",
    "load_nnfoundation",
    "preprocess_nnfoundation_ct",
    "preprocess_nnfoundation_mri",
]


if __name__ == "__main__":
    cnn, _, _ = load_nnfoundation_cnn()
    vit, _, _ = load_nnfoundation_vit()

    input = torch.rand(1, 1, 192, 192, 192).to(_default_device())
    with torch.no_grad():
        cnn_output = cnn(input)
        vit_output = vit(input)

    print("CNN output shape:", cnn_output.shape)
    print("ViT output shape:", vit_output.shape)
