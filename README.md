# Radiology feature extraction

`RadiologyProcessor` extracts features from radiology volumes listed in a CSV worklist. The first column contains a volume path for each row. Paths can point to a supported image file or, when DICOM series loading is enabled in the dataset, a series directory.

## Worklist with segmentations

When using `--use_segmentations`, provide a second column with the matching segmentation path for every volume. The rows are paired by position. By default, every row must have a segmentation path, and the file must be readable and align with the volume. For example:

```csv
volume_path,segmentation_path
/data/images/patient_001.nii.gz,/data/segmentations/patient_001.nii.gz
/data/images/patient_002.nii.gz,/data/segmentations/patient_002.nii.gz
```

Run extraction with `--use_segmentations`; segmentation discovery or matching by dataset metadata is not performed.

To allow cases without a segmentation, leave their second-column value empty or provide a path that is unavailable, then pass `--allow_missing_segmentations`. Those cases use the dataset's configured no-segmentation fallback; the default `FeatureExtractionDataset` uses `fallback_mode: center_crop` unless configured otherwise.

To visually review the image-mask pairs before extraction, run:

```bash
python scripts/review_segmentation_matches.py /data/volume_worklist_with_segmentations.csv
```

The script uses `VolumeSliceViewer` to step through every axial slice at 0.5 seconds per slice by default. Use `--interval` to change the delay. If an image and mask have different physical grids, it resamples the mask onto the image grid for display.

## Run

```bash
python feature_extraction.py \
  --worklist_csv /data/volume_worklist.csv \
  --save_dir /data/features/radiology \
  --fm_list merlin
```

For segmentation-aware preprocessing, add `--use_segmentations` and use the two-column worklist shown above. Use `--dataset_config_file` for dataset settings and `--model_config_file` for model-specific settings. `--save_type` selects HDF5 (`h5`, default) or PyTorch (`pt`) output.
