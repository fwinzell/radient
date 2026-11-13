"""Load Curia foundation model variants."""

from transformers import AutoImageProcessor, AutoModel


def _load(model_id, config, device):
    options = dict(config.get("from_pretrained", {}))
    model = AutoModel.from_pretrained(model_id, **options)
    processor_options = dict(config.get("processor_from_pretrained", {}))
    processor_options.setdefault("trust_remote_code", True)
    processor = AutoImageProcessor.from_pretrained(model_id, **processor_options)
    model.to(device)
    model.eval()
    return model, processor, "axial"


def load_curia(config, device):
    return _load(config.get("model_id", "raidium/curia"), config, device)


def load_curia_2(config, device):
    return _load(config.get("model_id", "raidium/curia-2"), config, device)
