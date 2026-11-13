"""Model name registry and common adapter interface."""

from importlib import import_module

_LOADERS = {
    "curia": ("curia", "load_curia"),
    "curia-2": ("curia", "load_curia_2"),
    "tap-ct-2d": ("tap_ct", "load_2d"),
    "tap-ct-3d": ("tap_ct", "load_3d"),
    "renalclip": ("renalclip", "load_renalclip"),
    "renal-clip": ("renalclip", "load_renalclip"),
    "merlin": ("merlin", "load_merlin"),
    "nnfoundation-cnn": ("nnfoundation", "load_nnfoundation"),
    "nnfoundation": ("nnfoundation", "load_nnfoundation"),
}


def available_models():
    return tuple(_LOADERS)


def load_foundation_model(name, config=None, device="cpu"):
    """Return ``(model, processor, mode)`` for a registered model.

    ``mode`` is ``"axial"`` for 2D slice encoders and ``"3d"`` for
    volume encoders. Model-specific options are passed to the adapter in
    ``config``.
    """
    try:
        module_name, function_name = _LOADERS[name.lower()]
    except (KeyError, AttributeError):
        raise ValueError(
            f"Unknown foundation model {name!r}. Available models: "
            f"{', '.join(available_models())}"
        ) from None
    loader = getattr(import_module(f".{module_name}", __package__), function_name)
    return loader(config or {}, device)
