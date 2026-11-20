#!/usr/bin/env python3
"""Display a 3D CT volume with its segmentation in VolumeSliceViewer.

Both inputs must be 3D MetaImage ``.mha`` files. If the segmentation uses a
different physical grid, it is resampled onto the CT grid using nearest-neighbor
interpolation.

Example:
    python scripts/view_volume_segmentation.py /data/ct.nii.gz /data/seg.nii.gz
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import SimpleITK as sitk

# When this file is run directly, Python puts scripts/ (not the repository
# root) on sys.path. Add the root so the local radient package is importable
# without requiring an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from radient.dataset.viewer import VolumeSliceViewer


def read_volume(path: Path) -> sitk.Image:
    if not path.exists():
        raise FileNotFoundError(path)
    
    image = sitk.ReadImage(str(path))

    if image.GetDimension() != 3:
        raise ValueError(f"Expected a 3D volume at {path}, got {image.GetDimension()}D")
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ct", type=Path, help="Path to a 3D CT .mha file")
    parser.add_argument("segmentation", type=Path, help="Path to the corresponding segmentation .mha file")
    parser.add_argument(
        "--window-level", type=float, default=50,
        help="CT display window level in HU (default: 50, soft tissue)",
    )
    parser.add_argument(
        "--window-width", type=float, default=500,
        help="CT display window width in HU (default: 500, soft tissue)",
    )
    args = parser.parse_args()
    if args.window_width <= 0:
        parser.error("--window-width must be greater than 0")

    ct = read_volume(args.ct.expanduser())
    segmentation = read_volume(args.segmentation.expanduser())
    # Apply a standard soft-tissue CT window and scale to [0, 1] for display.
    # Raw HU values span a much wider range than matplotlib's useful display
    # range, which can make most anatomy appear nearly black.
    ct_array = sitk.GetArrayFromImage(ct).astype(np.float32)
    low = args.window_level - args.window_width / 2
    ct_array = np.clip((ct_array - low) / args.window_width, 0, 1)
    segmentation_array = sitk.GetArrayFromImage(segmentation)

    # Center-pad the smaller array on each axis so both arrays have the same
    # [D, H, W] shape. Padding the already-windowed CT keeps the added border
    # black; segmentation background remains label 0.
    target_shape = tuple(max(c, s) for c, s in zip(ct_array.shape, segmentation_array.shape))
    if ct_array.shape != segmentation_array.shape:
        print(
            f"CT shape {ct_array.shape} and segmentation shape "
            f"{segmentation_array.shape} differ; zero-padding to {target_shape}."
        )

        def centered_padding(array):
            return tuple(
                ((target - current) // 2, (target - current + 1) // 2)
                for current, target in zip(array.shape, target_shape)
            )

        ct_array = np.pad(ct_array, centered_padding(ct_array), mode="constant")
        segmentation_array = np.pad(
            segmentation_array,
            centered_padding(segmentation_array),
            mode="constant",
            constant_values=0,
        )

    viewer = VolumeSliceViewer(
        image=ct_array,
        segmentation=segmentation_array,
        vmin=0,
        vmax=1,
    )
    viewer.show(title=args.ct.name)


if __name__ == "__main__":
    main()


"""
Example:
    python scripts/view_volume_segmentation.py /data/ct.nii.gz /data/seg.nii.gz

    python scripts/view_volume_segmentation.py \
    "/Volumes/PA_CPGARCHIVE/projects/multimodal_baseline/kidney/datasets/TCGA-KIRC/radiology/TCGA-KIRC-mha/TCGA-CW-5587/04-20-1997-NA-Abdomen020APRoutineAbdomenPelvis Adult-51162/4.000000-Biopsy Abd  2.4  B30s-58797/1.3.6.1.4.1.14519.5.2.1.3344.4004.178589701623300717946179551162-0/ef073c52-fe04-47bf-b15a-0e5c65936360.mha" \
    "/Volumes/PA_CPGARCHIVE/projects/multimodal_baseline/kidney/datasets/TCGA-KIRC/radiology/onco_segmentations/KIRC_1_3_6_1_4_1_14519_5_2_1_3344_4004_232318176582171184290906958797.nrrd" 
"""
