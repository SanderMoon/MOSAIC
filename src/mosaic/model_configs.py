from dataclasses import dataclass, field
from typing import Optional

from peft.tuners.lora.config import LoraConfig

"""
Documentation on the model configurations can be found in the Mosaic documentation at mosaic/docs/model_configs.md.
"""


@dataclass
class GenericConfig:
    embed_dim: int = 1024


@dataclass
class VisionConfig(GenericConfig):
    input_dim: int = 192
    method: str = "vit"
    pooling_strategy: str = "attentional"
    attentional_pool_vector: bool = True
    attn_pooler_vector_heads: int = 8
    attentional_pool_matrix: bool = True
    attn_pooler_matrix_length: int = 128
    attn_pooler_matrix_heads: int = 8
    extra_config: dict = field(default_factory=dict)


@dataclass
class TextConfig(GenericConfig):
    context_length: int = 77
    load_pretrained: bool = True
    hf_model_name: str = "microsoft/biogpt"
    hf_tokenizer_name: str = "microsoft/biogpt"
    load_pretrained_model_start_layer: Optional[int] = None
    load_pretrained_model_end_layer: Optional[int] = None
    vocab_size: int = 42384
    heads: int = 12
    layers: int = 12
    attn_pooler_vector_heads: int = 8
    freeze_base: bool = False
    freeze_embedding: bool = False
    lora: bool = False
    cls_token_id: Optional[int] = None
    lora_config: LoraConfig = field(
        default_factory=lambda: LoraConfig(
            r=4,
            lora_alpha=4,
            target_modules=["q_proj", "k_proj", "v_proj"],
            lora_dropout=0.0,
            bias="none",
        )
    )
    extra_config: dict = field(default_factory=dict)


@dataclass
class MultimodalConfig(GenericConfig):
    context_length: int = 77
    hf_model_name: str = "microsoft/biogpt"
    load_pretrained_model_start_layer: int = 0
    load_pretrained_model_end_layer: int = 12
    vocab_size: int = 42384
    heads: int = 12
    layers: int = 12
    cross_attention_type: str = "default"
    load_pretrained: bool = True
    freeze_base: bool = True
    freeze_cross_attn: bool = False
    extra_config: dict = field(default_factory=dict)
