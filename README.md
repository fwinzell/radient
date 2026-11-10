# CT feature extraction

`CTProcessor` extracts features from CT volumes using foundation models. Run it through `feature_extraction.py`; the input worklist is a CSV with a header and one CT path per row. Create one with `create_worklist.py` or provide an existing worklist.

## Supported encoders

- `curia`, `curia-2`, `tap-ct-2d`: axial slice features
- `tap-ct-3d`, `merlin`: whole-volume features
- `renalclip` (or `renal-clip`): kidney-focused 3D features; set `--renalclip_repo_path`

Use `--organ kidney` to load paired left/right crops with `KidneyFeatureDataset`. Kidney mode expects segmentation masks with kidney label `1` and lesion label `2`; configure mask matching with `--use_segmentations` and `--segmentation_dir`. Without kidney mode, `GenericTCIADataset` loads whole volumes and can optionally return matched segmentations.

## Run

```bash
python scripts/feature_encoders/radiology/feature_extraction.py \
  --img_dir /data/ct \
  --worklist_csv /data/ct_worklist.csv \
  --save_dir /data/features/radiology \
  --fm_list tap-ct-2d tap-ct-3d
```

Pass `--dataset_config_file scripts/feature_encoders/radiology/config/tcga_kidney.yaml` to load preprocessing settings from YAML. See `feature_extraction.py --help` for all options. `--save_type` selects HDF5 (`h5`, default) or PyTorch (`pt`) output; features are organized under model-specific `axial_features_*` or `3d_features_*` directories. Optional center-slice thumbnails are saved under `thumbnails/`.

For Slurm, edit paths and model options in `run_ct_fms.sh`, then submit with:

```bash
sbatch scripts/feature_encoders/radiology/run_ct_fms.sh
```
