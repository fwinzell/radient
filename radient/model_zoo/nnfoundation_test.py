import torch
import json
import torch.nn as nn

from huggingface_hub import hf_hub_download ,snapshot_download

from nnssl.architectures.get_network_from_plan import get_network_from_plans

from dynamic_network_architectures.architectures.primus import PrimusS, PrimusB, PrimusM, PrimusL, Primus
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet


IMAGE_EXTENSIONS = (".nii.gz", ".nii", ".mha", ".mhd", ".nrrd")
STAGE_CHANNELS = [32, 64, 128, 256, 320, 320]
DIVISOR = 32  # total downsampling of the encoder (5 stride-2 stages)


# ----------------------------------------------------------------------------- model


def load_cnn_encoder(checkpoint: str | None, device: torch.device) -> nn.Module:
    """ResEnc-L as in nnssl.architectures.architecture_registry.get_res_enc_l, with nnFoundationCNN weights."""
    n_stages = 6
    net = ResidualEncoderUNet(
        input_channels=1,
        n_stages=n_stages,
        features_per_stage=STAGE_CHANNELS,
        conv_op=nn.Conv3d,
        kernel_sizes=[[3, 3, 3]] * n_stages,
        strides=[[1, 1, 1]] + [[2, 2, 2]] * 5,
        n_blocks_per_stage=[1, 3, 4, 6, 6, 6],
        num_classes=1,
        n_conv_per_stage_decoder=[1] * 5,
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        deep_supervision=False,
    )
    if checkpoint is None:
        from huggingface_hub import hf_hub_download

        checkpoint = hf_hub_download("MIC-DKFZ/nnFoundationCNN", "checkpoint_final.pth")
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=False)["network_weights"]
    net.load_state_dict(state_dict, strict=True)
    encoder = net.encoder
    encoder.return_skips = True
    return encoder.to(device).eval()




"""
path = snapshot_download(
    "MIC-DKFZ/nnFoundationViT", 
    allow_patterns=["*.pth", "*.json"])


ckpt = torch.load(f"{path}/checkpoint_final.pth", map_location="cpu", weights_only=True)

plan = json.load(open(f"{path}/adaptation_plan.json", "r"))

print(plan)
arch_kwargs = dict(plan["architecture_plans"]["arch_kwargs"])
arch_kwargs['decoder_eva_depth'] = 0

from nnssl.architectures.evaMAE_module import EvaMAE


model = EvaMAE(**arch_kwargs, 
               do_up_projection=False)

arch_kwargs.pop("input_channels", None)
arch_kwargs.pop("output_channels", None)

network = get_network_from_plans(
    arch_class_name=plan["architecture_plans"]["arch_class_name"],
    arch_kwargs=arch_kwargs,
    arch_kwargs_req_import=[], #plan["architecture_plans"]["arch_kwargs_requiring_import"],
    input_channels=1,  # Should be different
    output_channels=2,
    allow_init=False,  # Always false
    deep_supervision=False,
)
print(network)

model.load_state_dict(ckpt['network_weights'], strict=False)
model.eval()
"""
device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_cnn_encoder(checkpoint=None, device=device)


input = torch.rand(1, 1, 192, 192, 192)
with torch.no_grad():
    output = model(input)

print("Input shape:", input.shape)
print("Output shape:", output[0].shape)


