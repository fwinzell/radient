import torch
import json
import torch.nn as nn
from einops import rearrange

from huggingface_hub import hf_hub_download ,snapshot_download

#from nnssl.architectures.get_network_from_plan import get_network_from_plans

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



INPUT_SHAPE = (192, 192, 192)
PATCH_EMBED_SIZE = (8, 8, 8)

def load_vit_encoder(checkpoint: str | None, device: torch.device) -> tuple[nn.Module, nn.Module]:
    """EvaMAE as in nnssl.architectures.architecture_registry.get_eva_mae, with nnFoundationViT weights."""
    model = Primus(
                input_channels=1,
                embed_dim=1056,
                num_classes=1,
                eva_depth=40,
                eva_numheads=16,
                input_shape=INPUT_SHAPE,
                patch_embed_size=PATCH_EMBED_SIZE,
                init_values=0.1,
                scale_attn_inner=True,
            )
    if checkpoint is None:
        from huggingface_hub import hf_hub_download

        checkpoint = hf_hub_download("MIC-DKFZ/nnFoundationViT", "checkpoint_final.pth")
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=False)["network_weights"]
    
    model.load_state_dict(state_dict, strict=False)
    eva = model.eva
    down_projection = model.down_projection
    return eva.to(device).eval(), down_projection.to(device).eval()


device = "cuda" if torch.cuda.is_available() else "cpu"
eva, down_projection = load_vit_encoder(checkpoint=None, device=torch.device(device))
unet = load_cnn_encoder(checkpoint=None, device=torch.device(device))

input = torch.rand(1, 1, 192, 192, 192).to(device)
with torch.no_grad():

    x = down_projection(input)
    x = rearrange(x, "b c w h d -> b (w h d) c")
    output = eva(x)
    
    #output = unet(input)


print("Input shape:", input.shape)
print("Output shape:", output[0].shape)


