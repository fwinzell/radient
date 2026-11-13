"""Visualize nnFoundationCNN feature maps with PCA (top-3 components shown as RGB).

For each encoder stage, a single PCA is fitted on the voxel-wise feature vectors of all modalities
of a case (so the same color means the same feature direction across sequences). The projection is
shown on an axial slice through the mask center, next to the input image, with the mask outlined.

Preprocessing is identical to extract_nnfoundation_features.py.

Usage:
    python3 scripts/visualize_features_pca.py data/radiology features/
    python3 scripts/visualize_features_pca.py data/radiology features/ --case IC_S01_P00005 --stages 0 1 2 3 4
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

from extract_nnfoundation_features import (
    DIVISOR,
    crop_window,
    find_cases,
    load_encoder,
    make_reference_grid,
    pad_to_multiple,
    resample,
    zscore,
)


def preprocess_case(files: dict[str, Path], modalities: list[str], reference: str, spacing: float, max_size: int):
    """Returns {modality: padded input (z, y, x)}, padded mask or None, valid-region map."""
    ref_key = reference if reference in files else modalities[0]
    grid = make_reference_grid(sitk.ReadImage(str(files[ref_key])), spacing)
    mask = resample(sitk.ReadImage(str(files["mask"])), grid, is_mask=True) if "mask" in files else None
    window = crop_window(grid.GetSize()[::-1], max_size, mask)

    images = {}
    for modality in modalities:
        image = resample(zscore(sitk.ReadImage(str(files[modality]))), grid, is_mask=False)
        images[modality], valid_slices = pad_to_multiple(image[window], DIVISOR)
    valid = np.zeros_like(next(iter(images.values())))
    valid[valid_slices] = 1
    mask_padded = pad_to_multiple(mask[window], DIVISOR)[0] if mask is not None else None
    return images, mask_padded, valid


def to_rgb(proj: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.clip((proj - lo) / np.maximum(hi - lo, 1e-8), 0, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--case", default=None, help="case id; default: all cases")
    parser.add_argument("--modalities", nargs="+", default=None)
    parser.add_argument("--reference", default="t2w")
    parser.add_argument("--mask-suffix", default="mask")
    parser.add_argument("--spacing", type=float, default=1.0)
    parser.add_argument("--max-size", type=int, default=192)
    parser.add_argument("--stages", nargs="+", type=int, default=[0, 1, 2, 3, 4], help="encoder stages 0-5")
    parser.add_argument("--slice", type=int, default=None, help="axial slice in model-input space; default: mask center")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-fit-voxels", type=int, default=20000, help="voxels per modality used to fit PCA")
    args = parser.parse_args()

    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = find_cases(args.input_dir, args.mask_suffix)
    if args.case is not None:
        cases = {args.case: cases[args.case]}
    encoder = load_encoder(args.checkpoint, device)
    rng = np.random.default_rng(0)

    for case_id, files in cases.items():
        modalities = args.modalities or sorted(k for k in files if k != "mask")
        images, mask, valid = preprocess_case(files, modalities, args.reference, args.spacing, args.max_size)

        if args.slice is not None:
            z = args.slice
        elif mask is not None and mask.any():
            z = int(np.round(np.argwhere(mask > 0)[:, 0].mean()))
        else:
            z = int(np.argwhere(valid > 0)[:, 0].mean())

        feats = {}
        for modality, image in images.items():
            with torch.no_grad():
                out = encoder(torch.from_numpy(image)[None, None].to(device))
            feats[modality] = [f[0].float().cpu() for f in out]

        n_rows, n_cols = len(modalities), 1 + len(args.stages)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 2.6 * n_rows), squeeze=False)
        extent = (0, valid.shape[2], valid.shape[1], 0)

        for col, stage in enumerate(args.stages, start=1):
            stride = 2**stage
            # valid region at this stage's resolution (exclude zero padding from the PCA fit)
            fmap_shape = feats[modalities[0]][stage].shape[1:]
            valid_s = F.adaptive_avg_pool3d(torch.from_numpy(valid)[None, None], fmap_shape)[0, 0].numpy() > 0.5

            fit = []
            for modality in modalities:
                vox = feats[modality][stage].numpy()[:, valid_s].T  # (n_voxels, channels)
                idx = rng.choice(len(vox), size=min(args.max_fit_voxels, len(vox)), replace=False)
                fit.append(vox[idx])
            fit = np.concatenate(fit)
            pca = PCA(n_components=3).fit(fit)
            fit_proj = pca.transform(fit)
            lo, hi = np.percentile(fit_proj, 1, axis=0), np.percentile(fit_proj, 99, axis=0)

            zs = min(z // stride, fmap_shape[0] - 1)
            for row, modality in enumerate(modalities):
                sl = feats[modality][stage].numpy()[:, zs]  # (C, y, x)
                rgb = to_rgb(pca.transform(sl.reshape(sl.shape[0], -1).T), lo, hi).reshape(*sl.shape[1:], 3)
                rgb[~valid_s[zs]] = 0
                ax = axes[row, col]
                ax.imshow(rgb, extent=extent, interpolation="nearest")
                if mask is not None:
                    ax.contour(mask[z], levels=[0.5], colors="white", linewidths=0.8)
                if row == 0:
                    ev = pca.explained_variance_ratio_.sum() * 100
                    ax.set_title(f"stage {stage} ({fmap_shape[1]}x{fmap_shape[2]})\nPC1-3: {ev:.0f}% var", fontsize=9)

        for row, modality in enumerate(modalities):
            ax = axes[row, 0]
            img = images[modality][z]
            vmin, vmax = np.percentile(img[valid[z] > 0], [1, 99])
            ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
            if mask is not None:
                ax.contour(mask[z], levels=[0.5], colors="red", linewidths=0.8)
            ax.set_ylabel(modality, fontsize=11)
            if row == 0:
                ax.set_title(f"input (z={z}, {args.spacing} mm)", fontsize=9)
        for ax in axes.flat:
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle(f"{case_id} - nnFoundationCNN feature PCA (RGB = PC1, PC2, PC3; fitted jointly per stage)", fontsize=11)
        fig.tight_layout()
        out_path = args.output_dir / f"pca_{case_id}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[{case_id}] wrote {out_path}")


if __name__ == "__main__":
    main()
