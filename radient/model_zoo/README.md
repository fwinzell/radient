# Foundation model adapters

`RadiologyProcessor.run_all()` loads encoders through `registry.py`. Each
adapter exposes a loader that accepts `(config, device)` and returns:

```python
model, processor, mode
```

`mode` is `"axial"` for slice encoders or `"3d"` for volume encoders. Add a
model by implementing a loader module, then registering its public name in
`_LOADERS` in `registry.py`.

Supported names currently include `curia`, `curia-2`, `tap-ct-2d`,
`tap-ct-3d`, `renalclip`/`renal-clip`, `merlin`, `nnfoundation-cnn`, and
`nnfoundation-vit`.

Model-specific options are grouped in the processor's `model_configs`
dictionary. For example:

```python
model_configs = {
    "renalclip": {
        "repo_path": "/path/to/RenalCLIP",
        "init_kwargs": {"mode": "finetune"},
    },
    "curia": {
        "model_id": "raidium/curia",
        "from_pretrained": {},
        "processor_from_pretrained": {"trust_remote_code": True},
    },
}
```

The `from_pretrained`, `processor_from_pretrained`, and `init_kwargs` keys are
adapter conventions; each model adapter documents and interprets its own
options.

nnFoundation has two loaders in `nnfoundation.py`:
`load_nnfoundation_cnn()` and `load_nnfoundation_vit()`. They download the
official checkpoints from Hugging Face and return `(model, preprocessor,
"3d")`, matching the model-zoo loader interface. Use the registry names
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
