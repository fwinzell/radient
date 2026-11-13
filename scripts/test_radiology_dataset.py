#!/usr/bin/env python3
"""Smoke-test GenericTCIADataset with image files or DICOM series directories.

Examples:
  python scripts/test_radiology_dataset.py /data/case1/ct.nii.gz --modality CT
  python scripts/test_radiology_dataset.py /data/case1/mr_series --modality MRI --sequence T2
  python scripts/test_radiology_dataset.py --manifest cohort.csv --image-column image_path \
      --patient-column patient_id --modality-column modality

Manifest image paths may point to a 3D image file or a directory containing exactly
one DICOM series. This is a loading/geometry check; modality-specific intensity
conversion (for example PET SUV conversion) is intentionally not performed.
"""
import argparse
import traceback

import pandas as pd

from radient.dataset.generic import GenericRadiologyDataset



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="Image files or one-series DICOM directories")
    parser.add_argument("--manifest", help="CSV manifest containing one image path per row")
    parser.add_argument("--image-column", default="image_path")
    parser.add_argument("--segmentation-column")
    parser.add_argument("--patient-column")
    parser.add_argument("--study-column")
    parser.add_argument("--series-column")
    parser.add_argument("--modality-column")
    parser.add_argument("--sequence-column")
    parser.add_argument("--paired-image-column")
    parser.add_argument("--modality", help="Label for all positional paths, e.g. CT, MRI, PET")
    parser.add_argument("--sequence", help="Sequence label for all positional paths, e.g. T1, T2")
    parser.add_argument("--orientation", default="RAS")
    args = parser.parse_args()

    if args.manifest:
        frame = pd.read_csv(args.manifest)
        if args.modality and not args.modality_column:
            frame["_modality"] = args.modality
            args.modality_column = "_modality"
        if args.sequence and not args.sequence_column:
            frame["_sequence"] = args.sequence
            args.sequence_column = "_sequence"
        dataset = GenericRadiologyDataset(
            frame,
            orientation=args.orientation,
            image_path_column=args.image_column,
            segmentation_path_column=args.segmentation_column,
            patient_id_column=args.patient_column,
            study_id_column=args.study_column,
            series_id_column=args.series_column,
            modality_column=args.modality_column,
            sequence_column=args.sequence_column,
            paired_image_path_column=args.paired_image_column,
        )
    else:
        if not args.paths:
            parser.error("Provide image paths or --manifest")
        frame = pd.DataFrame({"image_path": args.paths})
        if args.modality:
            frame["modality"] = args.modality
        if args.sequence:
            frame["sequence"] = args.sequence
        dataset = GenericRadiologyDataset(
            frame,
            image_path_column="image_path",
            modality_column="modality" if args.modality else None,
            sequence_column="sequence" if args.sequence else None,
            segmentation_column=None,
            orientation=args.orientation,
        )

    failures = 0
    for idx in range(len(dataset)):
        try:
            sample = dataset[idx]
            meta = sample["metadata"]
            print(
                f"OK [{idx}] pid={sample['pid']} sid={sample['sid']} "
                f"series={sample['barcode']} modality={meta['modality']} "
                f"sequence={meta['sequence']} shape_zyx={meta['shape_zyx']} "
                f"spacing_xyz={meta['spacing']} origin={meta['origin']} "
                f"segmentation={meta['has_segmentation']}"
            )
            img = sample["image"]
            #viewer = VolumeSliceViewer(image=img)
        except Exception as exc:
            failures += 1
            print(f"\nFAIL [{idx}] path={dataset.paths[idx]}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    print(f"Finished: {len(dataset) - failures} passed, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
