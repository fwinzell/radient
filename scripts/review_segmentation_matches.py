#!/usr/bin/env python3
"""Step through volumes and their paired segmentations from a CSV worklist.

The first two CSV columns must contain the volume path and corresponding
segmentation path. Each axial slice is displayed for 0.5 seconds by default
using the project's VolumeSliceViewer.
Supported inputs are SimpleITK-readable 3D files and directories containing one
DICOM series.

Example:
    python scripts/review_segmentation_matches.py /data/worklist_with_segmentations.csv
    python scripts/review_segmentation_matches.py /data/worklist_with_segmentations.csv --interval 0.5
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from radient.dataset.viewer import VolumeSliceViewer


def read_volume(path: Path) -> sitk.Image:
    """Read one image file or one DICOM series directory."""
    if not path.exists():
        raise FileNotFoundError(path)

    if not path.is_dir():
        image = sitk.ReadImage(str(path))
    else:
        series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(path))
        if not series_ids:
            raise ValueError(f"No DICOM series found in {path}")
        if len(series_ids) != 1:
            raise ValueError(
                f"Found {len(series_ids)} DICOM series in {path}; provide a directory containing one series"
            )
        filenames = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
            str(path), series_ids[0]
        )
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(filenames)
        image = reader.Execute()

    if image.GetDimension() != 3:
        raise ValueError(f"Expected a 3D volume at {path}, got {image.GetDimension()}D")
    return image


def same_geometry(left: sitk.Image, right: sitk.Image) -> bool:
    """Return whether two volumes use the same physical voxel grid."""
    return (
        left.GetSize() == right.GetSize()
        and np.allclose(left.GetSpacing(), right.GetSpacing(), atol=1e-4)
        and np.allclose(left.GetOrigin(), right.GetOrigin(), atol=1e-4)
        and np.allclose(left.GetDirection(), right.GetDirection(), atol=1e-4)
    )


def load_pair(volume_path: Path, segmentation_path: Path):
    image = read_volume(volume_path)
    segmentation = read_volume(segmentation_path)

    if not same_geometry(image, segmentation):
        print(
            "  Note: image and segmentation grids differ; resampling the mask "
            "onto the image grid for review."
        )
        segmentation = sitk.Resample(
            segmentation,
            image,
            sitk.Transform(3, sitk.sitkIdentity),
            sitk.sitkNearestNeighbor,
            0,
            segmentation.GetPixelID(),
        )

    image_array = sitk.GetArrayFromImage(image)
    segmentation_array = sitk.GetArrayFromImage(segmentation)
    if image_array.shape != segmentation_array.shape:
        raise ValueError(
            f"Image and mask shapes still differ after alignment: "
            f"{image_array.shape} vs {segmentation_array.shape}"
        )
    return image_array, segmentation_array


def review_worklist(worklist_path: Path, interval: float) -> int:
    worklist = pd.read_csv(worklist_path)
    if worklist.shape[1] < 2:
        raise ValueError(
            "Worklist must have at least two columns: volume path first and "
            "matching segmentation path second."
        )
    if worklist.empty:
        raise ValueError("Worklist contains no cases")

    failures = 0

    for row_number, row in enumerate(worklist.itertuples(index=False, name=None), start=2):
        volume_value, segmentation_value = row[0], row[1]
        if pd.isna(volume_value) or not str(volume_value).strip():
            print(f"Skipping CSV row {row_number}: volume path is empty")
            failures += 1
            continue
        if pd.isna(segmentation_value) or not str(segmentation_value).strip():
            print(f"Skipping CSV row {row_number}: segmentation path is empty")
            failures += 1
            continue

        volume_value = str(volume_value).replace("/data/pa_cpgarchive/", "/Volumes/PA_CPGARCHIVE/")
        segmentation_value = str(segmentation_value).replace("/data/pa_cpgarchive/", "/Volumes/PA_CPGARCHIVE/")
        volume_path = Path(str(volume_value)).expanduser()
        segmentation_path = Path(str(segmentation_value)).expanduser()
        print(
            f"Case {row_number - 1}/{len(worklist)} (CSV row {row_number})\n"
            f"  Volume: {volume_path}\n"
            f"  Segmentation: {segmentation_path}"
        )

        try:
            image, segmentation = load_pair(volume_path, segmentation_path)
        except Exception as exc:
            failures += 1
            print(f"  Failed to load pair: {type(exc).__name__}: {exc}")
            continue

        viewer = VolumeSliceViewer(
            image=image,
            segmentation=segmentation,
            figsize=(12, 6),
        )
        viewer.show(
            autoplay_interval=interval,
            title=f"Case {row_number - 1}/{len(worklist)}: {volume_path.name}",
        )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("worklist", type=Path, help="CSV with volume and segmentation paths in its first two columns")
    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="Seconds to display each slice (default: 0.5)",
    )
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be greater than 0")
    failures = review_worklist(args.worklist, args.interval)
    print(f"Review finished with {failures} case(s) that could not be reviewed.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
