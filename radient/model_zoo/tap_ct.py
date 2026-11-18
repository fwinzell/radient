"""Load TAP-CT 2D and 3D encoders."""

from transformers import AutoImageProcessor, AutoModel


def _load(default_model_id, config, device):
    model_id = config.get("model_id", default_model_id)
    options = dict(config.get("from_pretrained", {}))
    options.setdefault("trust_remote_code", True)
    model = AutoModel.from_pretrained(model_id, **options)
    processor = None
    if config.get("load_processor", True):
        processor_options = dict(config.get("processor_from_pretrained", {}))
        processor_options.setdefault("trust_remote_code", True)
        processor = AutoImageProcessor.from_pretrained(model_id, **processor_options)
    model.to(device)
    model.eval()
    return model, processor


def load_2d(config, device):
    return _load("fomofo/tap-ct-b-2d", config, device)


def load_3d(config, device):
    return _load("fomofo/tap-ct-b-3d", config, device)
