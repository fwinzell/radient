"""Load the external RenalCLIP model."""

import sys
from pathlib import Path


def load_renalclip(config, device):
    repo_path = config.get("repo_path")
    if repo_path is None:
        raise ValueError("Set model_configs['renalclip']['repo_path'] to use RenalCLIP.")
    repo_path = Path(repo_path).expanduser().resolve()
    if not repo_path.exists():
        raise FileNotFoundError(f"RenalCLIP repository path does not exist: {repo_path}")

    for candidate in (str(repo_path.parent), str(repo_path)):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)

    # RenalCLIP's repository has generic module names which can collide with
    # similarly named modules already imported by a host application.
    for cache_key in (
        "utils", "utils.util", "utils.logger", "utils.data_util",
        "models", "models.RenalModel", "RenalCLIP.models.RenalModel",
    ):
        sys.modules.pop(cache_key, None)

    from RenalCLIP.models.RenalModel import RenalModel

    model = RenalModel(**config.get("init_kwargs", {"mode": "finetune"}))
    model.to(device)
    model.eval()
    return model, None, "3d"
