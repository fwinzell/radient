"""Extract nnFoundationCNN encoder features from 3D MRI volumes.

Runs on the existing docker image (Python 3.11, torch, SimpleITK, dynamic_network_architectures,
huggingface_hub) -- nnssl itself does not need to be installed.

Expected input layout (one file per case and sequence, plus an optional mask):

    <input_dir>/<case_id>_<modality>.(mha|nii.gz|nii|nrrd)
    e.g. IC_S01_P00005_t2w.mha, IC_S01_P00005_adc.mha, IC_S01_P00005_mask.nii.gz

Per case, every modality is
  1. z-scored over the whole image (as in nnssl pre-training),
  2. resampled onto one shared grid (the reference modality's field of view, isotropic --spacing),
  3. cropped to at most --max-size voxels per axis (centered on the mask, if any) and padded to a
     multiple of 32,
  4. passed separately through the encoder (the model was pre-trained on single-channel inputs).

From each of the 6 encoder stages two vectors are pooled:
  * global: mean over the whole (unpadded) volume
  * mask:   mean over the mask region (only if a mask is present)

Outputs in <output_dir>:
  * <case_id>.npz               all pooled vectors per modality and stage (+ preprocessing metadata)
  * features_global.csv         one row per (case, modality), all stages concatenated (1120 dims)
  * features_mask.csv           same, mask-pooled (only if masks were found)

Usage:
    python3 scripts/extract_nnfoundation_features.py data/radiology features/
    python3 scripts/extract_nnfoundation_features.py data/radiology features/ --modalities t2w adc hbv \
        --spacing 1.0 --checkpoint /shared/checkpoint_final.pth
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from torch import nn
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet

IMAGE_EXTENSIONS = (".nii.gz", ".nii", ".mha", ".mhd", ".nrrd")
STAGE_CHANNELS = [32, 64, 128, 256, 320, 320]
DIVISOR = 32  # total downsampling of the encoder (5 stride-2 stages)


# ----------------------------------------------------------------------------- model


def load_encoder(checkpoint: str | None, device: torch.device) -> nn.Module:
    """ResEnc-L as in nnssl.architectures.architecture_registry.get_res_enc_l, with nnFoundationCNN weights."""
    n_stages = 6
    net = ResidualEncoderUNet(
        input_channels=1,
        n_stages=n_stages,
        features_per_stage=STAGE_CHANNELS,
        conv_op=nn.Conv3d,
        kernel_sizes=[[3, 3, 3]] * n_stages,
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
    if checkpoint is None:
        from huggingface_hub import hf_hub_download

        checkpoint = hf_hub_download("MIC-DKFZ/nnFoundationCNN", "checkpoint_final.pth")
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=False)["network_weights"]
    net.load_state_dict(state_dict, strict=True)
    encoder = net.encoder
    encoder.return_skips = True
    return encoder.to(device).eval()


# ----------------------------------------------------------------------------- data


def strip_extension(name: str) -> str | None:
    for ext in IMAGE_EXTENSIONS:
        if name.endswith(ext):
            return name[: -len(ext)]
    return None


def find_cases(input_dir: Path, mask_suffix: str) -> dict[str, dict[str, Path]]:
    """Group files as {case_id: {modality_or_'mask': path}} using the last '_' as separator."""
    cases: dict[str, dict[str, Path]] = defaultdict(dict)
    for path in sorted(input_dir.iterdir()):
        stem = strip_extension(path.name)
        if stem is None or "_" not in stem:
            continue
        case_id, suffix = stem.rsplit("_", 1)
        cases[case_id]["mask" if suffix == mask_suffix else suffix] = path
    return dict(cases)


def zscore(image: sitk.Image) -> sitk.Image:
    """Whole-image z-score, matching nnssl's ZScoreNormalization (use_mask_for_norm=False)."""
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    arr = (arr - arr.mean()) / max(float(arr.std()), 1e-8)
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(image)
    return out


def make_reference_grid(reference: sitk.Image, spacing: float) -> sitk.Image:
    """Grid with the reference image's orientation and physical extent, at isotropic `spacing`."""
    old_size = np.array(reference.GetSize())
    old_spacing = np.array(reference.GetSpacing())
    new_size = np.maximum(np.round(old_size * old_spacing / spacing), 1).astype(int)
    grid = sitk.Image([int(s) for s in new_size], sitk.sitkFloat32)
    grid.SetDirection(reference.GetDirection())
    grid.SetSpacing([spacing] * 3)
    # keep the physical corner of the first voxel aligned (origin is a voxel center)
    direction = np.array(reference.GetDirection()).reshape(3, 3)
    corner = np.array(reference.GetOrigin()) - direction @ (old_spacing / 2)
    grid.SetOrigin(tuple(corner + direction @ (np.full(3, spacing) / 2)))
    return grid


def resample(image: sitk.Image, grid: sitk.Image, is_mask: bool) -> np.ndarray:
    interp = sitk.sitkLinear
    out = sitk.Resample(image, grid, sitk.Transform(), interp, 0.0, sitk.sitkFloat32)
    arr = sitk.GetArrayFromImage(out)  # (z, y, x) -- same axis order nnssl used in pre-training
    return (arr > 0.5).astype(np.float32) if is_mask else arr


def crop_window(shape: tuple[int, ...], max_size: int, mask: np.ndarray | None) -> tuple[slice, ...]:
    """Window of at most max_size per axis, centered on the mask's center of mass (or the volume center)."""
    if mask is not None and mask.any():
        center = np.round(np.argwhere(mask > 0).mean(axis=0)).astype(int)
    else:
        center = np.array(shape) // 2
    slices = []
    for c, s in zip(center, shape):
        size = min(s, max_size)
        start = int(np.clip(c - size // 2, 0, s - size))
        slices.append(slice(start, start + size))
    return tuple(slices)


def pad_to_multiple(arr: np.ndarray, divisor: int) -> tuple[np.ndarray, tuple[slice, ...]]:
    """Zero-pad (= the mean after z-scoring, as in nnssl) and return the slices of the original content."""
    pads, valid = [], []
    for s in arr.shape:
        total = (-s) % divisor
        pads.append((total // 2, total - total // 2))
        valid.append(slice(total // 2, total // 2 + s))
    return np.pad(arr, pads, mode="constant", constant_values=0), tuple(valid)


# ----------------------------------------------------------------------------- features


def pool_features(
    feats: list[torch.Tensor], valid: np.ndarray, mask: np.ndarray | None
) -> tuple[list[np.ndarray], list[np.ndarray] | None]:
    """Weighted average of each stage's feature map; weights are area-downsampled valid/mask maps."""
    def weighted_mean(f: torch.Tensor, w: np.ndarray) -> np.ndarray:
        w_t = torch.from_numpy(w)[None, None].to(f.device)
        w_t = F.adaptive_avg_pool3d(w_t, f.shape[2:])
        return ((f * w_t).sum(dim=(2, 3, 4)) / w_t.sum().clamp_min(1e-8))[0].float().cpu().numpy()

    global_vecs = [weighted_mean(f, valid) for f in feats]
    mask_vecs = [weighted_mean(f, valid * mask) for f in feats] if mask is not None and mask.any() else None
    return global_vecs, mask_vecs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--modalities", nargs="+", default=None, help="default: every non-mask suffix found")
    parser.add_argument("--reference", default="t2w", help="modality defining the shared grid / field of view")
    parser.add_argument("--mask-suffix", default="mask")
    parser.add_argument("--spacing", type=float, default=1.0, help="isotropic target spacing in mm")
    parser.add_argument("--max-size", type=int, default=192, help="max voxels per axis (pre-training patch: 192)")
    parser.add_argument("--checkpoint", default=None, help="local checkpoint_final.pth; default: Hugging Face")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", help="bf16/fp16 autocast on GPU (faster, less memory)")
    args = parser.parse_args()

    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = find_cases(args.input_dir, args.mask_suffix)
    if not cases:
        raise SystemExit(f"No images found in {args.input_dir}")
    print(f"Found {len(cases)} case(s) in {args.input_dir}; device={device}")

    encoder = load_encoder(args.checkpoint, device)
    columns = [f"s{s}_{c}" for s, n in enumerate(STAGE_CHANNELS) for c in range(n)]
    rows_global, rows_mask = [], []

    for case_id, files in cases.items():
        modalities = args.modalities or sorted(k for k in files if k != "mask")
        missing = [m for m in modalities if m not in files]
        if missing:
            print(f"[{case_id}] skipped: missing {missing}")
            continue
        ref_key = args.reference if args.reference in files else modalities[0]
        grid = make_reference_grid(sitk.ReadImage(str(files[ref_key])), args.spacing)

        mask = resample(sitk.ReadImage(str(files["mask"])), grid, is_mask=True) if "mask" in files else None
        window = crop_window(grid.GetSize()[::-1], args.max_size, mask)
        mask_crop = None
        if mask is not None:
            mask_crop, _ = pad_to_multiple(mask[window], DIVISOR)

        out = {"spacing": args.spacing, "reference": ref_key}
        for modality in modalities:
            image = resample(zscore(sitk.ReadImage(str(files[modality]))), grid, is_mask=False)
            image, valid_slices = pad_to_multiple(image[window], DIVISOR)
            valid = np.zeros_like(image)
            valid[valid_slices] = 1

            x = torch.from_numpy(image)[None, None].to(device)
            with torch.no_grad(), torch.autocast(device.type, enabled=args.amp and device.type == "cuda"):
                feats = encoder(x)
            global_vecs, mask_vecs = pool_features(feats, valid, mask_crop)

            for s, v in enumerate(global_vecs):
                out[f"{modality}_global_s{s}"] = v
            rows_global.append([case_id, modality, *np.concatenate(global_vecs)])
            if mask_vecs is not None:
                for s, v in enumerate(mask_vecs):
                    out[f"{modality}_mask_s{s}"] = v
                rows_mask.append([case_id, modality, *np.concatenate(mask_vecs)])

        out["input_shape_zyx"] = np.array(image.shape)
        np.savez(args.output_dir / f"{case_id}.npz", **out)
        print(f"[{case_id}] {modalities} -> input {tuple(image.shape)} (z, y, x) at {args.spacing} mm")

    for name, rows in (("features_global.csv", rows_global), ("features_mask.csv", rows_mask)):
        if rows:
            pd.DataFrame(rows, columns=["case_id", "modality", *columns]).to_csv(args.output_dir / name, index=False)
            print(f"Wrote {args.output_dir / name} ({len(rows)} rows x {len(columns)} features)")


if __name__ == "__main__":
    main()
