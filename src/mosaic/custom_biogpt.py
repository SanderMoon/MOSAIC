import logging
from dataclasses import asdict
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from peft.tuners.lora.config import LoraConfig
from torch.utils.checkpoint import checkpoint
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
)
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.biogpt.modeling_biogpt import (
    ACT2FN,
    BioGptAttention,
    BioGptConfig,
    BioGptPreTrainedModel,
)

from mosaic.model_configs import MultimodalConfig, TextConfig
from mosaic.utils import create_config_instance

logger = logging.getLogger(__name__)


class CustomBioGptConfig(BioGptConfig):
    """
    This class represents a custom configuration for a BioGpt model, extending the base BioGptConfig.

    It introduces additional parameters to control specific behaviors related to cross-attention,
    special tokens, pre-trained model loading, LoRA usage, and layer freezing.

    Args:
        cls_token_id (int, optional): The ID of the classification token.
        hf_model (str, optional): The name or path of a Hugging Face pre-trained BioGpt model.
        load_pretrained (bool, optional): Whether to load pre-trained weights from the original BioGpt model. Defaults to False.
        use_lora (bool, optional): Whether to use Low-Rank Adaptation (LoRA) for the language encoder. Defaults to False.
        lora_config (LoraConfig, optional): The configuration for LoRA if used.
        freeze_text_encoder (bool, optional): Whether to freeze the entire text encoder. Defaults to False.
        freeze_decoder_self_attn (bool, optional): Whether to freeze the self-attention layers in the decoder. Defaults to False.
        **kwargs: Additional keyword arguments passed to the base BioGptConfig.
    """

    def __init__(
        self,
        cls_token_id: Optional[int] = None,
        hf_model: Optional[str] = None,
        load_pretrained: Optional[bool] = False,
        use_lora: Optional[bool] = False,
        lora_config: Optional[LoraConfig] = None,
        freeze_text_encoder: Optional[bool] = False,
        freeze_decoder_self_attn: Optional[bool] = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cls_token_id = cls_token_id
        self.hf_model = hf_model
        self.load_pretrained = load_pretrained
        self.use_lora = use_lora
        self.lora_config = lora_config
        self.freeze_text_encoder = freeze_text_encoder
        self.freeze_decoder_self_attn = freeze_decoder_self_attn


class CustomBioGptDecoderLayer(nn.Module):
    """
    This class implements a custom decoder layer for a BioGpt model, extending the base decoder layer functionality.

    It provides support for standard cross-attention mechanism, configurable through
    the `CustomBioGptConfig`. Additionally, it handles layer normalization, self-attention,
    cross-attention, feed-forward networks, and residual connections.

    Args:
        config (CustomBioGptConfig): The configuration object for the custom BioGpt model.
    """

    def __init__(self, config: BioGptConfig):
        super().__init__()
        self.embed_dim = config.hidden_size

        self.self_attn = BioGptAttention(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            is_decoder=True,
        )

        self.cross_attn = BioGptAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            is_decoder=True,
        )

        self.dropout = config.hidden_dropout_prob
        self.activation_fn = ACT2FN[config.hidden_act]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.cross_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        self.fc1 = nn.Linear(self.embed_dim, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through the custom decoder layer.

        Args:
            hidden_states (torch.Tensor): The input hidden states of shape (batch_size, sequence_length, hidden_size).
            encoder_hidden_states (torch.Tensor, optional): The encoder hidden states for cross-attention,
                of shape (batch_size, encoder_sequence_length, hidden_size).
            attention_mask (torch.Tensor, optional): The attention mask for self-attention,
                of shape (batch_size, 1, 1, sequence_length).
            encoder_attention_mask (torch.Tensor, optional): The attention mask for cross-attention,
                of shape (batch_size, 1, 1, encoder_sequence_length).
            layer_head_mask (torch.Tensor, optional): The layer-wise head mask for self-attention,
                of shape (num_heads,).
            cross_attn_layer_head_mask (torch.Tensor, optional): The layer-wise head mask for cross-attention,
                of shape (num_heads,).
            past_key_value (Tuple[torch.Tensor], optional): Tuple containing past key and value tensors:
                (self_attn_past_key, self_attn_past_value, cross_attn_past_key, cross_attn_past_value).
            output_attentions (bool, optional): Whether to return attention weights. Defaults to False.
            use_cache (bool, optional): Whether to cache key and value states for future use. Defaults to True.

        Returns:
            Tuple[torch.FloatTensor, ...]: A tuple containing:
                - hidden_states (torch.FloatTensor): The output hidden states of shape
                    (batch_size, sequence_length, hidden_size).
                - self_attn_weights (torch.FloatTensor, optional): The self-attention weights,
                    if `output_attentions` is True.
                - cross_attn_weights (torch.FloatTensor, optional): The cross-attention weights,
                    if `output_attentions` is True and `encoder_hidden_states` is provided.
                - present_key_value (Tuple[torch.Tensor], optional): The present key and value states for
                    self-attention and cross-attention, if `use_cache` is True.
        """
        # Initialize present_key_value tuple for caching
        present_key_value = ()

        # Split past_key_value into self-attention and cross-attention components
        if past_key_value is not None:
            self_attn_past_key_value = past_key_value[
                :2
            ]  # (self_attn_past_key, self_attn_past_value)
            cross_attn_past_key_value = past_key_value[
                2:
            ]  # (cross_attn_past_key, cross_attn_past_value)
        else:
            self_attn_past_key_value = None
            cross_attn_past_key_value = None

        # Self-Attention Block
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self-Attention
        hidden_states, self_attn_weights, self_attn_present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )

        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states

        # Cross-Attention Block
        cross_attn_weights = None
        cross_attn_present_key_value = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.cross_attn_layer_norm(hidden_states)

            # Cross-Attention
            hidden_states, cross_attn_weights, cross_attn_present_key_value = (
                self.cross_attn(
                    hidden_states=hidden_states,
                    key_value_states=encoder_hidden_states,
                    past_key_value=cross_attn_past_key_value,
                    attention_mask=encoder_attention_mask,
                    layer_head_mask=cross_attn_layer_head_mask,
                    output_attentions=output_attentions,
                )
            )

            hidden_states = nn.functional.dropout(
                hidden_states, p=self.dropout, training=self.training
            )
            hidden_states = residual + hidden_states

        # Feed Forward Network
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.activation_dropout, training=self.training
        )
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )

        hidden_states = residual + hidden_states

        # Prepare outputs
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)
            if cross_attn_weights is not None:
                outputs += (cross_attn_weights,)

        if use_cache:
            # Combine present key values from self-attention and cross-attention
            present_key_value = (
                self_attn_present_key_value + cross_attn_present_key_value
                if cross_attn_present_key_value is not None
                else self_attn_present_key_value
            )
            outputs += (present_key_value,)

        return outputs


class MultimodalDecoderBioGPTModel(BioGptPreTrainedModel):
    """
    This class represents a multimodal decoder-based BioGpt model, extending the base BioGptPreTrainedModel.

    It utilizes custom decoder layers (`CustomBioGptDecoderLayer`) to process input embeddings, incorporating
    mechanisms for self-attention, cross-attention, layer normalization,
    and feed-forward networks.

    Args:
        config (BioGptConfig): The configuration object for the BioGpt model.
    """

    def __init__(self, config: BioGptConfig):
        super().__init__(config)
        self.config = config
        self.layerdrop = config.layerdrop
        self.dropout = config.hidden_dropout_prob
        self.embed_dim = config.hidden_size
        self.padding_idx = config.pad_token_id

        self.layers = nn.ModuleList(
            [CustomBioGptDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.layer_norm = nn.LayerNorm(self.embed_dim)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        """
        Forward pass through the multimodal decoder BioGpt model.

        Args:
            attention_mask (torch.FloatTensor, optional): The attention mask for self-attention, of shape
                (batch_size, sequence_length).
            encoder_attention_mask (torch.FloatTensor, optional): The attention mask for cross-attention,
                of shape (batch_size, encoder_sequence_length).
            head_mask (torch.FloatTensor, optional): The head mask for attention, of shape
                (num_layers, num_heads).
            cross_attn_head_mask (torch.FloatTensor, optional): The head mask for cross-attention, of shape
                (num_layers, num_heads).
            inputs_embeds (torch.FloatTensor): The input embeddings of shape (batch_size, sequence_length, hidden_size).
            encoder_hidden_states (torch.FloatTensor, optional): The encoder hidden states for cross-attention,
                of shape (batch_size, encoder_sequence_length, hidden_size).
            past_key_values (Tuple[Tuple[torch.Tensor]], optional): The past key and value states for
                self-attention and cross-attention, a tuple of tuples of tensors.
            use_cache (bool, optional): Whether to cache key and value states for future use. Defaults to
                the value specified in the configuration.
            output_attentions (bool, optional): Whether to return attention weights. Defaults to the
                value specified in the configuration.
            output_hidden_states (bool, optional): Whether to return all hidden states. Defaults to
                the value specified in the configuration.
            return_dict (bool, optional): Whether to return a BaseModelOutputWithPastAndCrossAttentions
                object. Defaults to the value specified in the configuration.

        Returns:
            Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]: Either a tuple containing the output
                hidden states, past key values, all hidden states, self-attention weights, and cross-attention
                weights, or a BaseModelOutputWithPastAndCrossAttentions object containing these outputs.
        """

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # Ensure inputs_embeds is provided
        if inputs_embeds is None:
            raise ValueError("You have to specify inputs_embeds")

        input_shape = inputs_embeds.size()[:-1]

        # past_key_values_length
        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )

        # Prepare attention masks
        if attention_mask is None:
            attention_mask = torch.ones(
                (
                    inputs_embeds.shape[0],
                    inputs_embeds.shape[1] + past_key_values_length,
                ),
                dtype=torch.bool,
                device=inputs_embeds.device,
            )
        elif attention_mask.shape[1] != past_key_values_length + input_shape[1]:
            raise ValueError(
                f"The provided attention mask has length {attention_mask.shape[1]}, but its length should be "
                f"{past_key_values_length + input_shape[1]} (sum of the lengths of current and past inputs)"
            )

        # Convert attention masks to 4D
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        if encoder_attention_mask is not None:
            encoder_attention_mask = _prepare_4d_attention_mask(
                encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[1]
            )

        hidden_states = inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        # Process decoder layers
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                assert all_hidden_states is not None
                all_hidden_states += (hidden_states,)

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(
                            *inputs,
                            output_attentions=output_attentions,
                            use_cache=use_cache,
                        )

                    return custom_forward

                layer_outputs = checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    encoder_hidden_states,
                    attention_mask,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    (
                        cross_attn_head_mask[idx]
                        if cross_attn_head_mask is not None
                        else None
                    ),
                    past_key_value,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=head_mask[idx] if head_mask is not None else None,
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx]
                        if cross_attn_head_mask is not None
                        else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            assert (
                layer_outputs is not None
            ), "Output from decoder layer or checkpoint should not be None"

            # Extract outputs
            hidden_states = layer_outputs[0]

            if use_cache:
                assert (
                    next_decoder_cache is not None
                ), "next_decoder_cache must be a tuple when use_cache is True"
                current_layer_cache = layer_outputs[-1]
                assert (
                    current_layer_cache is not None
                ), "Cache component should not be None"  # Optional check
                next_decoder_cache += (current_layer_cache,)

            if output_attentions:
                assert all_self_attns is not None, "all_self_attns should not be None"
                self_attn_weights = layer_outputs[1]
                all_self_attns += (self_attn_weights,)
                if encoder_hidden_states is not None:
                    assert (
                        all_cross_attentions is not None
                    ), "all_cross_attentions should not be None"
                    cross_attn_weights = layer_outputs[2]
                    assert (
                        cross_attn_weights is not None
                    ), "cross_attn_weights should not be None"
                    all_cross_attentions += (cross_attn_weights,)

        # Add last hidden state
        if output_hidden_states:
            assert all_hidden_states is not None, "all_hidden_states should not be None"
            all_hidden_states += (hidden_states,)

        hidden_states = self.layer_norm(hidden_states)

        next_cache = next_decoder_cache if use_cache and next_decoder_cache else None
        if not return_dict:
            outputs = tuple(
                v
                for v in [
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_self_attns,
                    all_cross_attentions,
                ]
                if v is not None
            )
            return outputs

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


def biogpt_config_adapter(text_config: TextConfig) -> BioGptConfig:
    """
    Adapts a TextConfig object to a BioGptConfig object for seamless configuration conversion.

    This function converts a `TextConfig` instance into a `BioGptConfig` instance by mapping
    relevant attributes and handling any additional arguments provided in `extra_args`.
    It ensures that only valid `BioGptConfig` fields are included in the final configuration.

    Args:
        text_config (TextConfig): The source TextConfig object to be adapted.

    Returns:
        BioGptConfig: The resulting BioGptConfig object created from the adapted configuration.
    """

    # Convert TextConfig to a dictionary
    config_dict = asdict(text_config)

    # Remove extra_args from the main dictionary
    extra_args = config_dict.pop("extra_config", {})

    # Map standard TextConfig attribute names to BioGptConfig attribute names
    attribute_mapping = {
        "embed_dim": "hidden_size",
        "layers": "num_hidden_layers",
        "heads": "num_attention_heads",
    }

    # Create a new dictionary with mapped attributes
    mapped_config = {}
    for text_attr, biogpt_attr in attribute_mapping.items():
        if text_attr in config_dict:
            mapped_config[biogpt_attr] = config_dict[text_attr]

    # Add remaining attributes from config_dict that are not in the mapping
    for key, value in config_dict.items():
        if key not in attribute_mapping:
            mapped_config[key] = value

    # Merge the mapped config with extra_args
    # extra_args will override standard config if there are conflicts
    merged_config = {**mapped_config, **extra_args}

    # Create BioGptConfig instance
    return create_config_instance(BioGptConfig, merged_config)


def biogpt_custom_config_adapter(
    multimodal_config: MultimodalConfig,
) -> CustomBioGptConfig:
    """
    Adapts a MultimodalConfig object to a CustomBioGptConfig object for seamless configuration conversion.

    This function converts a `MultimodalConfig` instance into a `CustomBioGptConfig` instance by mapping
    relevant attributes and handling any additional arguments.
    It ensures that the specific requirements of `CustomBioGptConfig` are met.

    Args:
        multimodal_config (MultimodalConfig): The source MultimodalConfig object to be adapted.

    Returns:
        CustomBioGptConfig: The resulting CustomBioGptConfig object created from the adapted configuration.
    """

    # Convert MultimodalConfig to a dictionary
    config_dict = asdict(multimodal_config)

    # Remove extra_args from the main dictionary
    extra_args = config_dict.pop("extra_config", {})

    # Map MultimodalConfig attribute names to CustomBioGptConfig attribute names
    attribute_mapping = {
        "embed_dim": "hidden_size",
        "layers": "num_hidden_layers",
        "heads": "num_attention_heads",
        "hf_model_name": "hf_model",
        "load_pretrained": "load_pretrained",
        "freeze_mhsa": "freeze_decoder_self_attn",
    }

    custom_config_dict = {}
    for multimodal_attr, custom_attr in attribute_mapping.items():
        if multimodal_attr in config_dict:
            custom_config_dict[custom_attr] = config_dict[multimodal_attr]

    # Handle attributes that don't have a direct mapping
    custom_config_dict["freeze_text_encoder"] = config_dict.get("freeze_proj", False)

    # Add any extra args that might be in the MultimodalConfig but not explicitly mapped
    for key, value in config_dict.items():
        if key not in attribute_mapping and key not in custom_config_dict:
            custom_config_dict[key] = value

    # Merge the mapped config with extra_args
    # extra_args will override standard config if there are conflicts
    custom_config_dict = {**custom_config_dict, **extra_args}

    # Create CustomBioGptConfig, passing all arguments
    return CustomBioGptConfig(**custom_config_dict)


def load_pretrained_biogpt_weights(
    target_model: nn.Module,
    weights_path: str,
    start_layer: int | None = 0,
    end_layer: int | None = None,
):
    """
    Loads pre-trained BioGpt weights into a target model, optionally specifying a layer range.

    This function loads pre-trained weights from a checkpoint file at `weights_path` into a
    target BioGpt model. It allows for selective loading of weights from specific layers
    using `start_layer` and `end_layer`.

    Args:
        target_model: The BioGpt model to load weights into.
        weights_path (str): The path to the checkpoint file containing the pre-trained weights.
        start_layer (int, optional): The starting layer index (inclusive) for loading weights. Defaults to 0.
        end_layer (int, optional): The ending layer index (exclusive) for loading weights.
            If None, loads weights for all layers from `start_layer` onwards.
    """

    source_dict = torch.load(weights_path, map_location="cpu")
    target_dict = target_model.state_dict()

    # Load all weights if no layer range is specified
    if not start_layer and not end_layer:
        # Create a new dictionary for the renamed keys
        new_source_dict = {}
        for name, param in source_dict.items():
            if name.startswith("biogpt."):
                new_name = name.replace("biogpt.", "")
            else:
                new_name = name
            new_source_dict[new_name] = param
        # Load the state dict with the new keys
        target_model.load_state_dict(new_source_dict, strict=False)
        return

    if not start_layer:
        start_layer = 0
        logger.warning("No start layer specified, defaulting to 0")
    for name, param in source_dict.items():
        if name.startswith("biogpt.layers."):
            layer_num = int(name.split(".")[2])
            if start_layer <= layer_num < (end_layer or float("inf")):
                new_name = name.replace(
                    f"biogpt.layers.{layer_num}", f"layers.{layer_num - start_layer}"
                )
                new_name = new_name.replace("biogpt.", "")
                if new_name in target_dict:
                    target_dict[new_name].copy_(param)
        elif name.startswith("biogpt.") and name.replace("biogpt.", "") in target_dict:
            target_dict[name.replace("biogpt.", "")].copy_(param)
        else:
            print(f"Skipping loading weights of layer {name}")

    target_model.load_state_dict(target_dict, strict=True)
