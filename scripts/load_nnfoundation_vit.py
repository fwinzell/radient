"""Load the nnFoundationViT pretraining model from its local files.

Edit the two paths in the main block and run this Python file in an
environment where nnssl and its dependencies are installed. It recreates the
same EvaMAE architecture used by nnssl pretraining, then restores the complete
``network_weights`` state dictionary.
"""

import json
from pathlib import Path

import torch


def load_nnfoundation_vit(checkpoint_path, adaptation_plan_path, device="cpu"):
    """Return the restored nnssl EvaMAE model and its adaptation plan."""
    try:
        from nnssl.architectures.evaMAE_module import EvaMAE
    except ImportError as exc:
        raise ImportError(
            "Install the MIC-DKFZ/nnssl repository (nnFoundation branch) and "
            "its dependencies in this Python environment."
        ) from exc

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    adaptation_plan_path = Path(adaptation_plan_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not adaptation_plan_path.is_file():
        raise FileNotFoundError(f"Adaptation plan not found: {adaptation_plan_path}")

    # The checkpoint itself includes this plan. Prefer it so architecture
    # parameters always match the weights; the separate JSON is a fallback.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    plan = checkpoint.get("nnssl_adaptation_plan")
    if plan is None:
        with adaptation_plan_path.open() as stream:
            plan = json.load(stream)

    architecture = plan["architecture_plans"]
    if architecture["arch_class_name"] != "PrimusX":
        raise ValueError(
            "Expected the nnFoundationViT PrimusX plan, got "
            f"{architecture['arch_class_name']!r}"
        )
    if "network_weights" not in checkpoint:
        raise ValueError("Checkpoint is missing its 'network_weights' state dictionary")

    # This matches nnssl's BaseEvaMAETrainerPrimusX.build_architecture_and_adaptation_plan.
    model = EvaMAE(**architecture["arch_kwargs"])
    model.load_state_dict(checkpoint["network_weights"], strict=True)
    model.to(device)
    model.eval()
    return model, plan


if __name__ == "__main__":
    checkpoint_file = Path("/path/to/checkpoint_final.pth")
    adaptation_plan_file = Path("/path/to/adaptation_plan.json")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, plan = load_nnfoundation_vit(
        checkpoint_file,
        adaptation_plan_file,
        device=device,
    )
    n_parameters = sum(parameter.numel() for parameter in model.parameters())
    patch = plan["architecture_plans"]["arch_kwargs"]["input_shape"]
    print(f"Restored nnFoundationViT PrimusX on {device}")
    print(f"Parameters: {n_parameters:,}; pretraining input shape: {tuple(patch)}")
