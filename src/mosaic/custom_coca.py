"""This module implements a custom CoCa (Contrastive Captioners) model
It borrows components from the OpenClip library found here: https://github.com/mlfoundations/open_clip
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn import LayerNorm
from torch.nn import functional as F
from transformers import (
    AlbertModel,
    BeamSearchScorer,
    BertModel,
    DistilBertModel,
    EosTokenCriteria,
    LogitsProcessorList,
    MaxLengthCriteria,
    MinLengthLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    RobertaModel,
    StoppingCriteriaList,
    TopKLogitsWarper,
    TopPLogitsWarper,
    XLNetModel,
)
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

from mosaic.attentional_pooling import AttentionalPooling
from mosaic.model_builders import (
    build_multimodal_tower,
    build_text_tower,
    build_vision_tower,
)
from mosaic.model_configs import MultimodalConfig, TextConfig, VisionConfig
from mosaic.utils import _token_to_tensor

GENERATION_TYPES = {
    "top_k": TopKLogitsWarper,
    "top_p": TopPLogitsWarper,
    "beam_search": "beam_search",
}
_has_transformers = True


class VisionEncoder(nn.Module):
    """
    This class represents a vision encoder module, responsible for processing visual input
    and generating encoded visual representations.

    It utilizes a vision tower (`build_vision_tower`) to extract features from the input,
    and optionally employs attentional pooling mechanisms for generating compact
    visual embeddings.

    Args:
        vision_cfg (VisionConfig): The configuration object for the vision encoder.
    """

    def __init__(self, vision_cfg: VisionConfig):
        super().__init__()
        self.config = vision_cfg

        # Core vision feature extractor
        self.vision_encoder = build_vision_tower(vision_cfg)

        # Optional attentional pooling layers
        self.visual_pooling_strategy = vision_cfg.pooling_strategy
        if self.visual_pooling_strategy == "attentional":
            if vision_cfg.attentional_pool_vector:
                # Parameters for vector pooling
                visual_pooling_dim = vision_cfg.embed_dim
                visual_pooling_heads = vision_cfg.attn_pooler_vector_heads
                self.learnable_query = nn.Parameter(
                    torch.randn(1, 1, vision_cfg.embed_dim)
                )
                self.vector_pooling = AttentionalPooling(
                    visual_pooling_dim, visual_pooling_heads
                )

            if vision_cfg.attentional_pool_matrix:
                # Parameters for matrix pooling
                visual_pooling_matrix_n = vision_cfg.attn_pooler_matrix_length
                visual_pooling_dim = vision_cfg.embed_dim
                visual_pooling_heads = vision_cfg.attn_pooler_matrix_heads
                self.learnable_matrix = nn.Parameter(
                    torch.randn(1, visual_pooling_matrix_n, visual_pooling_dim)
                )
                self.matrix_pooling = AttentionalPooling(
                    visual_pooling_dim, visual_pooling_heads
                )

    def forward(
        self, vision_input: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ):
        """
        Forward pass through the vision encoder.

        Args:
            vision_input (torch.Tensor): The input visual data.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - vision_encoder_output: The encoded visual representations from the vision tower.
                - vision_embedding: The compact visual embedding (if attentional pooling is used).
        """

        # Process input through the vision tower
        vision_encoder_output = self.vision_encoder(vision_input, attention_mask)

        # Default assignment to avoid unbound variable warning
        vision_embedding = None

        # Apply attentional pooling if configured
        if self.visual_pooling_strategy == "attentional":
            if self.config.attentional_pool_vector:
                vision_embedding = self.vector_pooling(
                    vision_encoder_output, self.learnable_query, attention_mask
                )
            if self.config.attentional_pool_matrix:
                vision_encoder_output = self.matrix_pooling(
                    vision_encoder_output, self.learnable_matrix, attention_mask
                )
                # if matrix pooling is used, the output is the pooled representation and does not require a mask anymore.
                attention_mask = torch.ones(
                    vision_encoder_output.shape[0],
                    vision_encoder_output.shape[1],
                    dtype=torch.float,
                    device=vision_encoder_output.device,
                )
        elif self.visual_pooling_strategy == "cls":
            # Extract the CLS token (first token) as the visual embedding
            vision_embedding = vision_encoder_output[:, 0, :]
            vision_encoder_output = vision_encoder_output[:, 1:, :]
        else:
            vision_embedding = vision_encoder_output[:, 0, :]
            vision_encoder_output = vision_encoder_output[:, 1:, :]
            attention_mask = torch.ones(
                vision_encoder_output.shape[0],
                vision_encoder_output.shape[1],
                dtype=torch.float,
                device=vision_encoder_output.device,
            )

        # Ensure vision_embedding is never unbound
        if vision_embedding is None:
            vision_embedding = vision_encoder_output[:, 0, :]

        # squeeze the output to remove the extra dimension
        vision_embedding = vision_embedding.squeeze(1)

        return vision_encoder_output, vision_embedding, attention_mask


class TextEncoder(nn.Module):
    def __init__(self, text_cfg: "TextConfig"):
        """
        Args:
            text_cfg (TextConfig): Configuration for the text encoder.
            model_type (str): Type of the model, e.g., 'biogpt', 'bert'.
        """
        super().__init__()
        self.config = text_cfg

        # Core text feature extractor
        self.text_encoder = build_text_tower(text_cfg)

        # Determine if the model has a built-in CLS token
        self.has_cls = self._check_if_model_has_cls()

        # If the model lacks a CLS token, add a custom one
        if not self.has_cls:
            visual_pooling_dim = text_cfg.embed_dim
            visual_pooling_heads = text_cfg.attn_pooler_vector_heads
            self.learnable_query = nn.Parameter(torch.randn(1, 1, text_cfg.embed_dim))
            self.vector_pooling = AttentionalPooling(
                visual_pooling_dim, visual_pooling_heads
            )
        else:
            self.cls_embedding = None  # Not used for models with built-in CLS

    def _check_if_model_has_cls(self) -> bool:
        """
        Checks if the text_encoder model uses a built-in CLS token by verifying
        if it's an instance of models known to have a CLS token.

        Returns:
            bool: True if the model has a built-in CLS token, False otherwise.
        """
        # Tuple of model classes that inherently have a CLS token
        models_with_cls = (
            BertModel,
            RobertaModel,
            XLNetModel,
            AlbertModel,
            DistilBertModel,
        )

        return isinstance(self.text_encoder, models_with_cls)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if attention_mask is None:
            attention_mask = torch.ones(
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=torch.float,
                device=input_ids.device,
            )

            # Pass through the text encoder with the input IDs
        text_encoder_output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        if not self.has_cls:
            # Extract the custom CLS token (first token)
            text_cls_token = self.vector_pooling(
                text_encoder_output.last_hidden_state,
                self.learnable_query,
                attention_mask,
            )
            text_cls_token = text_cls_token.squeeze(1)
            other_tokens = text_encoder_output.last_hidden_state
        else:
            # Extract the built-in CLS token (first token)
            text_cls_token = text_encoder_output.last_hidden_state[:, 0, :]
            other_tokens = text_encoder_output.last_hidden_state[:, 1:, :]

        return other_tokens, text_cls_token


class MultimodalDecoder(nn.Module):
    """
    This class represents a multimodal decoder module, responsible for processing
    encoded multimodal representations (likely from a vision and text encoder)
    and generating output logits for text generation or other tasks.

    It utilizes a multimodal tower (`build_multimodal_tower`) for further processing
    of the encoded inputs. It also includes a final layer normalization and a
    projection layer to map the decoder's output to the vocabulary space.

    Args:
        multimodal_cfg (MultimodalConfig): The configuration object for the multimodal decoder.
    """

    def __init__(self, multimodal_cfg: MultimodalConfig):
        super().__init__()
        self.config = multimodal_cfg

        # Core multimodal processing tower
        self.multimodal_decoder = build_multimodal_tower(multimodal_cfg)

        # Final layer normalization
        self.ln_final = LayerNorm(self.config.embed_dim)

        # Projection layer to vocabulary space
        self.text_projection = nn.Linear(self.config.embed_dim, self.config.vocab_size)

    def forward(
        self,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        """
        Forward pass through the multimodal decoder.

        Args:
            inputs_embeds (torch.FloatTensor, optional): The input embeddings from the text encoder
            encoder_hidden_states (torch.FloatTensor, optional): The encoded representations
                from the encoder (e.g., vision encoder).
            attention_mask (torch.FloatTensor, optional): The attention mask for self-attention.

        Returns:
            BaseModelOutputWithPastAndCrossAttentions: An object containing the logits
                representing the model's output over the vocabulary.
        """

        decoder_output = self.multimodal_decoder(
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
        )

        last_hidden_state = decoder_output.last_hidden_state
        last_hidden_state = self.ln_final(last_hidden_state)
        logits = self.text_projection(last_hidden_state)

        return logits


@dataclass
class CoCaOutput:
    """
    This dataclass represents the output of a CoCa (Contrastive Captioners) model.

    It encapsulates the following:

    Attributes:
        text_embedding (torch.FloatTensor): The encoded representation of the input text.
        vision_embedding (torch.FloatTensor): The encoded representation of the input image or visual data.
        logits (torch.FloatTensor): The output logits from the model, typically used for
            calculating loss or generating predictions.
        labels (torch.Tensor, optional): The ground truth labels associated with the input,
            used for training or evaluation.
    """

    text_embedding: torch.FloatTensor
    vision_embedding: torch.FloatTensor
    logits: torch.FloatTensor
    logit_scale: torch.FloatTensor
    labels: Optional[torch.Tensor] = None


class CustomCoCa(nn.Module):
    """
    This class represents a custom CoCa (Contrastive Captioners) model, combining
    a vision encoder, a text encoder, and a multimodal decoder.

    It facilitates the processing of both visual and textual inputs, generating
    encoded representations for each modality and then passing them through
    a multimodal decoder to produce output logits.

    Args:
        vision_cfg (VisionConfig): The configuration object for the vision encoder.
        text_cfg (TextConfig): The configuration object for the text encoder.
        multimodal_cfg (MultimodalConfig): The configuration object for the multimodal decoder.
    """

    def __init__(
        self,
        vision_cfg: VisionConfig,
        text_cfg: TextConfig,
        multimodal_cfg: MultimodalConfig,
        init_logit_scale: float = np.log(1 / 0.07),
        init_logit_bias: Optional[float] = None,
    ):
        super().__init__()

        # Initialize the vision, text, and multimodal components
        self.vision_encoder = VisionEncoder(vision_cfg)
        self.text_encoder = TextEncoder(text_cfg)
        self.multimodal_decoder = MultimodalDecoder(multimodal_cfg)

        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)

    def _encode_image(self, vision_input, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones(
                vision_input.shape[:-1], dtype=torch.long, device=vision_input.device
            )
        return self.vision_encoder(vision_input, attention_mask)

    def forward(
        self,
        vision_input: torch.Tensor,
        text_input_ids: Optional[torch.LongTensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        image_feature_attention_mask: Optional[torch.Tensor] = None,
        image_latent: Optional[torch.Tensor] = None,
        image_embs: Optional[torch.Tensor] = None,
        output_labels: bool = True,
    ):
        """
        Forward pass through the custom CoCa model.

        Args:
            text_input_ids (torch.LongTensor, optional): The input token IDs for the text encoder.
            vision_input (torch.FloatTensor, optional): The input visual data for the vision encoder.
            output_labels (bool, optional): Whether to include labels in the output for
                teacher-forcing during training. Defaults to False.

            text_attention_mask (torch.FloatTensor, optional): The attention mask for the text input.
            image_feature_attention_mask (torch.FloatTensor, optional): The attention mask for the visual input.
            image_latent (torch.Tensor, optional): Precomputed visual latent representations.
            image_embs (torch.Tensor, optional): Precomputed visual embeddings.

        Returns:
            A dictionary containing:
            - text_features: Encoded text features from the text encoder.
            - image_features: Encoded image features from the vision encoder.
            - logits: Output logits from the multimodal decoder.
            - logit_scale: The scale factor for logits, used for scaling the output logits.
            - labels: Ground truth labels for teacher-forcing, if output_labels is True.

        """

        if image_feature_attention_mask is None:
            image_feature_attention_mask = torch.ones(
                vision_input.shape[0],
                vision_input.shape[1],
                dtype=torch.float,
                device=vision_input.device,
            )

        if image_latent is None or image_embs is None:
            # Encode visual input
            image_embs, image_latent, image_feature_attention_mask = self._encode_image(
                vision_input, image_feature_attention_mask
            )

        if text_input_ids is None:
            return {"image_features": image_latent, "image_embs": image_embs}

        try:
            # Encode textual input
            contextualized_text_tokens, text_embedding = self.text_encoder(
                input_ids=text_input_ids, attention_mask=text_attention_mask
            )
        except Exception as e:
            logging.error(f"Error in text encoder: {e}")
            logging.info(f"Text input ids: {text_input_ids}")
            logging.info(f"Text attention mask: {text_attention_mask}")
            raise e

        # Prepare labels for teacher-forcing if needed
        labels: Optional[torch.Tensor] = (
            text_input_ids[:, 1:] if output_labels else None
        )
        if output_labels:
            # Align text embeddings and logits with labels for teacher-forcing
            contextualized_text_tokens = contextualized_text_tokens[:, :-1]
            assert text_attention_mask is not None
            text_attention_mask = text_attention_mask[:, 1:]

        # Decode multimodal representations
        decoder_logits = self.multimodal_decoder(
            inputs_embeds=contextualized_text_tokens,
            encoder_hidden_states=image_embs,
            attention_mask=text_attention_mask,
            encoder_attention_mask=image_feature_attention_mask,
        )

        return {
            "text_features": text_embedding,
            "image_features": image_latent,
            "logits": decoder_logits,
            "logit_scale": self.logit_scale.exp(),
            "labels": labels,
        }

    def generate(
        self,
        image,
        text=None,
        seq_len=30,
        max_seq_len=77,
        temperature=1.0,
        generation_type="beam_search",
        top_p=0.1,  # keep tokens in the 1 - top_p quantile
        top_k=1,  # keeps the top_k most probable tokens
        pad_token_id=None,
        eos_token_id=None,
        sot_token_id=None,
        num_beams=6,
        num_beam_groups=3,
        min_seq_len=5,
        stopping_criteria=None,
        repetition_penalty=1.0,
        fixed_output_length=False,  # if True output.shape == (batch_size, seq_len)
        visual_attention_mask=None,
    ):
        # taking many ideas and components from HuggingFace GenerationMixin
        # https://huggingface.co/docs/transformers/main/en/main_classes/text_generation
        assert (
            _has_transformers
        ), "Please install transformers for generate functionality. `pip install transformers`."
        assert seq_len > min_seq_len, "seq_len must be larger than min_seq_len"
        device = image.device

        if visual_attention_mask is None:
            visual_attention_mask = torch.ones(
                image.shape[0], image.shape[1], dtype=torch.float, device=image.device
            )

        # expand attention_mask to num_beams to (num_beams * batch_size,  seq length)
        if generation_type == "beam_search":
            visual_attention_mask = torch.repeat_interleave(
                visual_attention_mask, num_beams, dim=0
            )

        with torch.no_grad():
            sot_token_id = _token_to_tensor(
                49406 if sot_token_id is None else sot_token_id, device=device
            )
            eos_token_id = _token_to_tensor(
                49407 if eos_token_id is None else eos_token_id, device=device
            )
            pad_token_id = self.pad_id if pad_token_id is None else pad_token_id
            logit_processor = LogitsProcessorList(
                [
                    MinLengthLogitsProcessor(min_seq_len, eos_token_id),
                    RepetitionPenaltyLogitsProcessor(repetition_penalty),
                ]
            )

            if stopping_criteria is None:
                stopping_criteria = [
                    MaxLengthCriteria(max_length=seq_len),
                    EosTokenCriteria(eos_token_id),
                ]
            stopping_criteria = StoppingCriteriaList(stopping_criteria)

            if generation_type == "beam_search":
                output = self._generate_beamsearch(
                    image_inputs=image,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    sot_token_id=sot_token_id,
                    num_beams=num_beams,
                    num_beam_groups=num_beam_groups,
                    min_seq_len=min_seq_len,
                    stopping_criteria=stopping_criteria,
                    logit_processor=logit_processor,
                    visual_attention_mask=visual_attention_mask,
                )
                if fixed_output_length and output.shape[1] < seq_len:
                    pad_len = seq_len - output.shape[1]
                    return torch.cat(
                        (
                            output,
                            torch.ones(
                                output.shape[0],
                                pad_len,
                                device=device,
                                dtype=output.dtype,
                            )
                            * pad_token_id,
                        ),
                        dim=1,
                    )
                return output

            elif generation_type == "top_p":
                logit_warper = GENERATION_TYPES[generation_type](top_p)
            elif generation_type == "top_k":
                logit_warper = GENERATION_TYPES[generation_type](top_k)
            else:
                raise ValueError(
                    f"generation_type has to be one of "
                    f"{'| ' + ' | '.join(list(GENERATION_TYPES.keys())) + ' |'}."
                )

            image_embs, image_latent, visual_attention_mask = self._encode_image(
                image, visual_attention_mask
            )

            if text is None:
                text = (
                    torch.ones((image.shape[0], 1), device=device, dtype=torch.long)
                    * sot_token_id
                )

            was_training = self.training
            num_dims = len(text.shape)

            if num_dims == 1:
                text = text[None, :]

            self.eval()
            out = text

            while True:
                x = out[:, -max_seq_len:]
                cur_len = x.shape[1]
                logits = self(
                    vision_input=image,
                    text_input_ids=x,
                    image_feature_attention_mask=visual_attention_mask,
                    image_latent=image_latent,
                    image_embs=image_embs,
                    output_labels=False,
                )["logits"][:, -1]
                mask = (out[:, -1] == eos_token_id) | (out[:, -1] == pad_token_id)
                sample = (
                    torch.ones((out.shape[0], 1), device=device, dtype=torch.long)
                    * pad_token_id
                )

                if mask.all():
                    if not fixed_output_length:
                        break
                else:
                    logits = logits[~mask, :]
                    filtered_logits = logit_processor(x[~mask, :], logits)
                    filtered_logits = logit_warper(x[~mask, :], filtered_logits)
                    probs = F.softmax(filtered_logits / temperature, dim=-1)

                    if cur_len + 1 == seq_len:
                        sample[~mask, :] = (
                            torch.ones((sum(~mask), 1), device=device, dtype=torch.long)
                            * eos_token_id
                        )
                    else:
                        sample[~mask, :] = torch.multinomial(probs, 1)

                out = torch.cat((out, sample), dim=-1)

                cur_len += 1

                if all(stopping_criteria(out, None)):
                    break

            if num_dims == 1:
                out = out.squeeze(0)

            self.train(was_training)
            return out

    def _generate_beamsearch(
        self,
        image_inputs,
        pad_token_id=None,
        eos_token_id=None,
        sot_token_id=None,
        num_beams=6,
        num_beam_groups=3,
        min_seq_len=5,
        stopping_criteria=None,
        logit_processor=None,
        logit_warper=None,
        visual_attention_mask=None,
    ):
        device = image_inputs.device
        batch_size = image_inputs.shape[0]
        image_inputs = torch.repeat_interleave(image_inputs, num_beams, dim=0)
        image_embs, image_latent, visual_attention_mask = self._encode_image(
            image_inputs, visual_attention_mask
        )

        input_ids = torch.ones(
            (batch_size * num_beams, 1), device=device, dtype=torch.long
        )
        input_ids = input_ids * sot_token_id
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=num_beams,
            device=device,
            num_beam_groups=num_beam_groups,
        )
        # instantiate logits processors
        logits_processor = (
            LogitsProcessorList(
                [MinLengthLogitsProcessor(min_seq_len, eos_token_id=eos_token_id)]
            )
            if logit_processor is None
            else logit_processor
        )

        num_beams = beam_scorer.num_beams
        num_beam_groups = beam_scorer.num_beam_groups
        num_sub_beams = num_beams // num_beam_groups
        batch_size = len(beam_scorer._beam_hyps) // num_beam_groups
        batch_beam_size, cur_len = input_ids.shape
        beam_indices = None

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        beam_scores = torch.full(
            (batch_size, num_beams), -1e9, dtype=torch.float, device=device
        )
        # initialise score of first beam of each group with 0 and the rest with 1e-9. This ensures that the beams in
        # the same group don't produce same tokens everytime.
        beam_scores[:, ::num_sub_beams] = 0
        beam_scores = beam_scores.view((batch_size * num_beams,))

        while True:
            # predicted tokens in cur_len step
            current_tokens = torch.zeros(
                batch_size * num_beams, dtype=input_ids.dtype, device=device
            )

            # indices which will form the beams in the next time step
            reordering_indices = torch.zeros(
                batch_size * num_beams, dtype=torch.long, device=device
            )

            # do one decoder step on all beams of all sentences in batch
            model_inputs = prepare_inputs_for_generation(
                input_ids=input_ids, image_inputs=image_inputs
            )

            outputs = self(
                vision_input=model_inputs["images"],
                text_input_ids=model_inputs["text"],
                image_feature_attention_mask=visual_attention_mask,
                image_latent=image_latent,
                image_embs=image_embs,
                output_labels=False,
            )

            for beam_group_idx in range(num_beam_groups):
                group_start_idx = beam_group_idx * num_sub_beams
                group_end_idx = min(group_start_idx + num_sub_beams, num_beams)
                group_size = group_end_idx - group_start_idx

                # indices of beams of current group among all sentences in batch
                batch_group_indices = []

                for batch_idx in range(batch_size):
                    batch_group_indices.extend(
                        [
                            batch_idx * num_beams + idx
                            for idx in range(group_start_idx, group_end_idx)
                        ]
                    )
                group_input_ids = input_ids[batch_group_indices]

                # select outputs of beams of currentg group only
                next_token_logits = outputs["logits"][batch_group_indices, -1, :]
                vocab_size = next_token_logits.shape[-1]

                next_token_scores_processed = logits_processor(
                    group_input_ids,
                    next_token_logits,
                    current_tokens=current_tokens,
                    beam_group_idx=beam_group_idx,
                )
                next_token_scores = next_token_scores_processed + beam_scores[
                    batch_group_indices
                ].unsqueeze(-1)
                next_token_scores = next_token_scores.expand_as(
                    next_token_scores_processed
                )

                # reshape for beam search
                next_token_scores = next_token_scores.view(
                    batch_size, group_size * vocab_size
                )

                next_token_scores, next_tokens = torch.topk(
                    next_token_scores, 2 * group_size, dim=1, largest=True, sorted=True
                )

                next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
                next_tokens = next_tokens % vocab_size

                # stateless
                process_beam_indices = (
                    sum(beam_indices, ()) if beam_indices is not None else None
                )
                beam_outputs = beam_scorer.process(
                    group_input_ids,
                    next_token_scores,
                    next_tokens,
                    next_indices,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    beam_indices=process_beam_indices,
                    group_index=beam_group_idx,
                )
                beam_scores[batch_group_indices] = beam_outputs["next_beam_scores"]
                beam_next_tokens = beam_outputs["next_beam_tokens"]
                beam_idx = beam_outputs["next_beam_indices"]

                input_ids[batch_group_indices] = group_input_ids[beam_idx]
                group_input_ids = torch.cat(
                    [group_input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)],
                    dim=-1,
                )
                current_tokens[batch_group_indices] = group_input_ids[:, -1]

                # (beam_idx // group_size) -> batch_idx
                # (beam_idx % group_size) -> offset of idx inside the group
                reordering_indices[batch_group_indices] = (
                    num_beams * torch.div(beam_idx, group_size, rounding_mode="floor")
                    + group_start_idx
                    + (beam_idx % group_size)
                )

            input_ids = torch.cat([input_ids, current_tokens.unsqueeze(-1)], dim=-1)

            # increase cur_len
            cur_len = cur_len + 1
            if beam_scorer.is_done or all(stopping_criteria(input_ids, None)):
                break

        final_beam_indices = sum(beam_indices, ()) if beam_indices is not None else None
        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=final_beam_indices,
        )
        return sequence_outputs["sequences"]

    def encode_text(self, input_ids, attention_mask=None):
        return self.text_encoder(input_ids, attention_mask)


def prepare_inputs_for_generation(input_ids, image_inputs, past=None, **kwargs):
    if past:
        input_ids = input_ids[:, -1].unsqueeze(-1)

    attention_mask = kwargs.get("attention_mask", None)
    position_ids = kwargs.get("position_ids", None)

    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
    else:
        position_ids = None
    return {
        "text": input_ids,
        "images": image_inputs,
        "past_key_values": past,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
    }


class CustomCLIP(nn.Module):
    """
    This class represents a custom CoCa (Contrastive Captioners) model, combining
    a vision encoder, a text encoder, and a multimodal decoder.

    It facilitates the processing of both visual and textual inputs, generating
    encoded representations for each modality and then passing them through
    a multimodal decoder to produce output logits.

    Args:
        vision_cfg (VisionConfig): The configuration object for the vision encoder.
        text_cfg (TextConfig): The configuration object for the text encoder.
        multimodal_cfg (MultimodalConfig): The configuration object for the multimodal decoder.
    """

    def __init__(
        self,
        vision_cfg: VisionConfig,
        text_cfg: TextConfig,
        init_logit_scale: float = np.log(1 / 0.07),
        init_logit_bias: Optional[float] = None,
    ):
        super().__init__()

        # Initialize the vision, text, and multimodal components
        self.vision_encoder = VisionEncoder(vision_cfg)
        self.text_encoder = TextEncoder(text_cfg)

        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)

    def _encode_image(self, vision_input, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones(
                vision_input.shape[:-1], dtype=torch.long, device=vision_input.device
            )
        return self.vision_encoder(vision_input, attention_mask)

    def forward(
        self,
        text_input_ids: Optional[torch.LongTensor] = None,
        vision_input: Optional[torch.FloatTensor] = None,
        text_attention_mask: Optional[torch.FloatTensor] = None,
        image_feature_attention_mask: Optional[torch.FloatTensor] = None,
        image_latent: Optional[torch.Tensor] = None,
        image_embs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor, BaseModelOutputWithPastAndCrossAttentions]:
        """
        Forward pass through the custom CoCa model.

        Args:
            text_input_ids (torch.LongTensor, optional): The input token IDs for the text encoder.
            vision_input (torch.FloatTensor, optional): The input visual data for the vision encoder.
            output_labels (bool, optional): Whether to include labels in the output for
                teacher-forcing during training. Defaults to False.

        Returns:
            CoCaOutput: An object containing the encoded representations of the input text and vision, and decoder output logits.
        """

        if image_feature_attention_mask is None:
            image_feature_attention_mask = torch.ones(
                vision_input.shape[0],
                vision_input.shape[1],
                dtype=torch.float,
                device=vision_input.device,
            )

        if image_latent is None or image_embs is None:
            # Encode visual input
            image_embs, image_latent, image_feature_attention_mask = self._encode_image(
                vision_input, image_feature_attention_mask
            )

        if text_input_ids is None:
            return {"image_features": image_latent, "image_embs": image_embs}

        # Encode textual input
        contextualized_text_tokens, text_embedding = self.text_encoder(
            input_ids=text_input_ids, attention_mask=text_attention_mask
        )

        return {
            "text_features": text_embedding,
            "image_features": image_latent,
            "logit_scale": self.logit_scale.exp(),
            "logits": None,
            "labels": None,
        }

    def encode_text(self, input_ids, attention_mask=None):
        return self.text_encoder(input_ids, attention_mask)
