from functools import partial

from peft.mapping import get_peft_model
from torch import nn
from transformers import BertConfig, BertModel
from transformers.models.biogpt.modeling_biogpt import BioGptModel

from mosaic.custom_biogpt import (
    MultimodalDecoderBioGPTModel,
    biogpt_config_adapter,
    biogpt_custom_config_adapter,
    load_pretrained_biogpt_weights,
)
from mosaic.custom_hipt_vit import VisionTransformerWSI
from mosaic.download_model import get_model_path
from mosaic.model_configs import MultimodalConfig, TextConfig, VisionConfig
from mosaic.perceiver_pytorch import PrismPerceiverEncoder
from simplified.pooler_encoder import PoolerOnlyEncoder


def build_text_tower(text_cfg: TextConfig) -> nn.Module:
    """
    Constructs the text encoder tower based on the provided configuration.

    Currently supports loading pretrained embeddings from 'microsoft/biogpt'
    and applying LoRA (Low-Rank Adaptation) if specified.

    Args:
        text_cfg (TextConfig): Configuration object for the text encoder.

    Returns:
        nn.Module: The constructed text encoder tower.

    Raises:
        ValueError: If unsupported configurations are encountered.
    """

    if not text_cfg.load_pretrained:
        raise ValueError("Only loading pretrained embeddings is currently supported")

    if text_cfg.hf_model_name == "microsoft/biogpt":
        config = biogpt_config_adapter(text_cfg)
        text_encoder = BioGptModel(config)
        weights_path = get_model_path(text_cfg.hf_model_name)
        load_pretrained_biogpt_weights(
            text_encoder,
            weights_path,
            text_cfg.load_pretrained_model_start_layer,
            text_cfg.load_pretrained_model_end_layer,
        )
    elif text_cfg.hf_model_name.startswith("bert"):
        if text_cfg.load_pretrained:
            text_encoder = BertModel.from_pretrained(text_cfg.hf_model_name)
        else:
            config = BertConfig.from_pretrained(text_cfg.hf_model_name)
            text_encoder = BertModel(config)
    else:
        raise ValueError(
            "Only microsoft/biogpt and bert models are currently supported for loading pretrained embeddings"
        )

    # Optionally freeze the base model parameters
    if text_cfg.freeze_base:
        for param in text_encoder.parameters():
            param.requires_grad = False

        if not text_cfg.freeze_embedding:
            embedding_layer = text_encoder.get_input_embeddings()
            for param in embedding_layer.parameters():
                param.requires_grad = True

    # Optionally apply LoRA
    if text_cfg.lora:
        text_encoder = get_peft_model(text_encoder, text_cfg.lora_config)

    return text_encoder


def build_vision_tower(vision_cfg: VisionConfig) -> nn.Module:
    """
    Constructs the vision encoder tower based on the provided configuration.

    Currently supports the 'vit' (Vision Transformer) method.

    Args:
        vision_cfg (VisionConfig): Configuration object for the vision encoder.

    Returns:
        nn.Module: The constructed vision encoder tower.

    Raises:
        ValueError: If unsupported vision methods are encountered.
    """

    if vision_cfg.method == "vit":
        vision_encoder = VisionTransformerWSI(
            **vision_cfg.extra_config,
            input_embed_dim=vision_cfg.input_dim,
            output_embed_dim=vision_cfg.embed_dim,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )
    elif vision_cfg.method == "pooler_only":
        image_features_dim = vision_cfg.input_dim
        common_dim = vision_cfg.embed_dim
        vision_encoder = PoolerOnlyEncoder(
            input_dim=image_features_dim, num_heads=8, output_dim=common_dim
        )
    elif vision_cfg.method == "perceiver":
        vision_encoder = PrismPerceiverEncoder(
            input_dim=vision_cfg.input_dim,
            output_dim=vision_cfg.embed_dim,
            **vision_cfg.extra_config,
        )
    else:
        raise ValueError(
            f"Unsupported vision method: {vision_cfg.method}. Supported methods are: 'vit', 'pooler_only', 'perceiver'."
        )

    return vision_encoder


def build_multimodal_tower(multimodal_cfg: MultimodalConfig) -> nn.Module:
    """
    Constructs the multimodal decoder tower based on the provided configuration.

    Currently supports loading pretrained embeddings from 'microsoft/biogpt'
    and selectively freezing layers (cross-attention or base layers).

    Args:
        multimodal_cfg (MultimodalConfig): Configuration object for the multimodal decoder.

    Returns:
        nn.Module: The constructed multimodal decoder tower.

    Raises:
        ValueError: If unsupported configurations are encountered.
    """

    if not multimodal_cfg.load_pretrained:
        raise ValueError("Only loading pretrained embeddings is currently supported")

    if multimodal_cfg.hf_model_name == "microsoft/biogpt":
        config = biogpt_custom_config_adapter(multimodal_cfg)
        model = MultimodalDecoderBioGPTModel(config)
        weights_path = get_model_path(multimodal_cfg.hf_model_name)
        load_pretrained_biogpt_weights(
            model,
            weights_path,
            multimodal_cfg.load_pretrained_model_start_layer,
            multimodal_cfg.load_pretrained_model_end_layer,
        )
    else:
        raise ValueError(
            "Only microsoft/biogpt is currently supported for loading pretrained embeddings"
        )

    # Optionally freeze cross-attention layers
    if multimodal_cfg.freeze_cross_attn:
        for layer in model.layers:
            if hasattr(layer, "cross_attn"):
                for param in layer.cross_attn.parameters():
                    param.requires_grad = False

    # Optionally freeze the base model layers
    if multimodal_cfg.freeze_base:
        for layer in model.layers:
            self_attn = layer.self_attn
            for param in self_attn.parameters():
                param.requires_grad = False

    return model
