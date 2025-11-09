from diffusers import UNet2DModel

from .mmdit import VideoDiTFutureFrameModel, VideoDiTModel
from .mmdit_sharing import VideoDiTFullSharingModel, VideoDiTModalitySharingModel
from .mmdit_split_attn import VideoDiTSplitAttnModel
from .uvit import VideoUViTModel


def get_model(cfg, latent_channels):
    if cfg.model.type == "hf":
        model = UNet2DModel(
            sample_size=cfg.model.image_size,  # the target image resolution
            in_channels=latent_channels,  # the number of input channels, 3 for RGB images
            out_channels=latent_channels,  # the number of output channels
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels=(
                224,
                224 * 2,
                224 * 3,
                224 * 4,
            ),  # (64, 64, 128, 128, 256, 256), # (128, 128, 256, 256, 512, 512), # the number of output channels for each UNet block
            down_block_types=(
                "DownBlock2D",  # a regular ResNet downsampling block
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                # "DownBlock2D",
                # "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                # "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",  # a regular ResNet upsampling block
                "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",
                "UpBlock2D",
                # "UpBlock2D",
                # "UpBlock2D",
            ),
        )
        return model
    elif cfg.model.type.lower() == "video_dit":
        dim_spatial = cfg.image_size // cfg.image_tokenizer.spatial_compression
        return VideoDiTModel(
            latent_channels,
            cfg.conditioning.num_past_frames,
            cfg.conditioning.num_future_frames,
            cfg.conditioning.num_past_latents,
            cfg.conditioning.num_future_latents,
            dim_spatial,
            dim_spatial,
            cfg.conditioning.dim_act,
            cfg.model.token_dim,
            cfg.model.patch_size,
            cfg.model.num_layers,
            cfg.model.num_heads,
            cfg.model.cfg_prob,
            discrete_time=cfg.use_discrete_time,
        )
    elif cfg.model.type.lower() == "video_dit_modality_sharing":
        dim_spatial = cfg.image_size // cfg.image_tokenizer.spatial_compression
        return VideoDiTModalitySharingModel(
            latent_channels,
            cfg.conditioning.num_past_frames,
            cfg.conditioning.num_future_frames,
            cfg.conditioning.num_past_latents,
            cfg.conditioning.num_future_latents,
            dim_spatial,
            dim_spatial,
            cfg.conditioning.dim_act,
            cfg.model.token_dim,
            cfg.model.patch_size,
            cfg.model.num_layers,
            cfg.model.num_heads,
            cfg.model.cfg_prob,
            discrete_time=cfg.use_discrete_time,
        )
    elif cfg.model.type.lower() == "video_dit_full_sharing":
        dim_spatial = cfg.image_size // cfg.image_tokenizer.spatial_compression
        return VideoDiTFullSharingModel(
            latent_channels,
            cfg.conditioning.num_past_frames,
            cfg.conditioning.num_future_frames,
            cfg.conditioning.num_past_latents,
            cfg.conditioning.num_future_latents,
            dim_spatial,
            dim_spatial,
            cfg.conditioning.dim_act,
            cfg.model.token_dim,
            cfg.model.patch_size,
            cfg.model.num_layers,
            cfg.model.num_heads,
            cfg.model.cfg_prob,
            discrete_time=cfg.use_discrete_time,
        )
    elif cfg.model.type.lower() == "video_dit_future_frame":
        dim_spatial = cfg.image_size // cfg.image_tokenizer.spatial_compression
        return VideoDiTFutureFrameModel(
            latent_channels,
            cfg.conditioning.num_past_frames,
            cfg.conditioning.num_future_frames,
            cfg.conditioning.num_past_latents,
            cfg.conditioning.num_future_latents,
            dim_spatial,
            dim_spatial,
            cfg.conditioning.dim_act,
            cfg.model.token_dim,
            cfg.model.patch_size,
            cfg.model.num_layers,
            cfg.model.num_heads,
            cfg.model.cfg_prob,
            discrete_time=cfg.use_discrete_time,
        )
    elif cfg.model.type.lower() == "video_dit_splitattn":
        dim_spatial = cfg.image_size // cfg.image_tokenizer.spatial_compression
        return VideoDiTSplitAttnModel(
            latent_channels,
            cfg.conditioning.num_past_frames,
            cfg.conditioning.num_future_frames,
            cfg.conditioning.num_past_latents,
            cfg.conditioning.num_future_latents,
            dim_spatial,
            dim_spatial,
            cfg.conditioning.dim_act,
            cfg.model.token_dim,
            cfg.model.patch_size,
            cfg.model.num_layers,
            cfg.model.num_heads,
            cfg.model.cfg_prob,
            discrete_time=cfg.use_discrete_time,
        )
    elif cfg.model.type.lower() == "video_uvit":
        dim_spatial = cfg.image_size // cfg.image_tokenizer.spatial_compression
        return VideoUViTModel(
            latent_channels,
            cfg.conditioning.num_past_frames,
            cfg.conditioning.num_future_frames,
            cfg.conditioning.num_past_latents,
            cfg.conditioning.num_future_latents,
            dim_spatial,
            dim_spatial,
            cfg.conditioning.dim_act,
            cfg.model.token_dim,
            cfg.model.patch_size,
            cfg.model.num_layers,
            cfg.model.num_heads,
            cfg.model.cfg_prob,
            discrete_time=cfg.use_discrete_time,
        )
    else:
        raise Exception("Unknown model type")
