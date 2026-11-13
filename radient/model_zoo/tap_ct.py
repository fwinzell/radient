"""Load TAP-CT 2D and 3D encoders."""

from transformers import AutoImageProcessor, AutoModel


def _load(default_model_id, mode, config, device):
    model_id = config.get("model_id", default_model_id)
    options = dict(config.get("from_pretrained", {}))
    options.setdefault("trust_remote_code", True)
    model = AutoModel.from_pretrained(model_id, **options)
    processor = None
    if mode == "axial" or config.get("load_processor", mode == "axial"):
        processor_options = dict(config.get("processor_from_pretrained", {}))
        processor_options.setdefault("trust_remote_code", True)
        processor = AutoImageProcessor.from_pretrained(model_id, **processor_options)
    model.to(device)
    model.eval()
    return model, processor, mode


def load_2d(config, device):
    return _load("fomofo/tap-ct-b-2d", "axial", config, device)


def load_3d(config, device):
    return _load("fomofo/tap-ct-b-3d", "3d", config, device)
