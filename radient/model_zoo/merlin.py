"""Load Merlin volume encoder."""


def load_merlin(config, device):
    from merlin import Merlin

    model = Merlin(**config.get("init_kwargs", {"ImageEmbedding": True}))
    model.to(device)
    model.eval()
    return model, None, "3d"
