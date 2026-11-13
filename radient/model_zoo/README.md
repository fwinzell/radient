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
`tap-ct-3d`, `renalclip`/`renal-clip`, `merlin`, and `nnfoundation-cnn`.

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

nnFoundationCNN constructs its ResEncL architecture and then loads the
checkpoint state dictionary. It accepts `checkpoint_path` for a local
`checkpoint_final.pth`, or downloads that file from Hugging Face by default.
The model expects one-channel 3D volumes and applies per-volume z-score
normalization by default. It does not resample the input. Loading requires
`dynamic-network-architectures>=0.4.4,<0.5` and `huggingface_hub` for automatic
download.
