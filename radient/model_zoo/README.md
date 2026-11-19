# Foundation model adapters

`RadiologyProcessor.run_all()` loads encoders through `registry.py`. Each
adapter exposes a loader that accepts `(config, device)` and returns:

```python
model, processor
```

Extraction and storage behavior are configured in `config/model_configs.yaml`. A new dataset and data loader are built for each model so its preprocessing stays isolated. Set `modality` to `ct` or `mri` in the dataset configuration; a model-level `modality` overrides it. If an adapter returns no preprocessor, the dataset uses a generic CT clip-and-scale transform or MRI per-volume z-score. Adapters that need source geometry can expose `preprocess_dataset(image, metadata, modality)`; the dataset applies that transform before generic spatial cropping.
`extraction_strategy` selects `axial` or `volume`, `save_strategy` selects
`axial` or `volume` output layout, and `model_input_strategy` selects a full
`volume` tensor or a `slice_batch` for models that preprocess a volume but
forward a batch of 2D slices. Add a model by implementing a loader module,
registering its public name in `_LOADERS` in `registry.py`, and adding its
settings to the YAML file.

Supported names currently include `curia`, `curia-2`, `tap-ct-2d`,
`tap-ct-3d`, `renalclip`/`renal-clip`, `merlin`, `nnfoundation-cnn`, and
`nnfoundation-vit`.

Model-specific loading options are grouped with extraction settings in the
processor's `model_configs` dictionary. Caller-provided values override the
defaults in `config/model_configs.yaml`. For example:

```python
model_configs = {
    "renalclip": {
        "repo_path": "/path/to/RenalCLIP",
        "init_kwargs": {"mode": "finetune"},
        "extraction_strategy": "volume",
        "save_strategy": "volume",
        "model_input_strategy": "volume",
    },
    "curia": {
        "model_id": "raidium/curia",
        "from_pretrained": {},
        "processor_from_pretrained": {"trust_remote_code": True},
        "extraction_strategy": "axial",
        "save_strategy": "axial",
    },
}
```

The `from_pretrained`, `processor_from_pretrained`, and `init_kwargs` keys are
adapter-specific loading options.

nnFoundation has two loaders in `nnfoundation.py`:
`load_nnfoundation_cnn()` and `load_nnfoundation_vit()`. They download the
official checkpoints from Hugging Face and return `(model, preprocessor)`. The
dataset applies their full-volume preprocessing to the source image. Use the registry names
`nnfoundation-cnn` and `nnfoundation-vit`; the optional model config key
`modality` selects the `ct` or `mri` preprocessor (defaults to `mri`). The CNN
returns its final ResEnc encoder feature map. The ViT returns the PrimusX EVA
patch-token embeddings, bypassing the segmentation up-projection/decoder. Both
preprocessors follow the released training intensity scheme: crop the
all-zero exterior, z-score each case, then center-crop or zero-pad to
192×192×192 for deterministic feature extraction. They do not resample voxel
spacing. CT and MRI use the same z-score scheme. Set
`HF_HOME` or `HF_HUB_CACHE` before Python starts to place downloads in a
different cache directory. Loading requires
`dynamic-network-architectures>=0.4.4,<0.5` and `huggingface_hub`.

Merlin's dataset preprocessor reorients to RAS, resamples to 1.5×1.5×3 mm,
scales CT values from [-1000, 1000] to [0, 1], then center-crops or pads to
160×224×224 in tensor order [D,H,W].
