"""
Wrapper for huggingface transformer models and attention pooling.
Beamsearch genetation code is taken from open_clip:
https://github.com/mlfoundations/open_clip
"""

from typing import Optional

import torch
import torch.nn as nn
from torch.nn import LayerNorm
from torch.nn import functional as F
from transformers import (
    AlbertModel,
    BertModel,
    DistilBertModel,
    RobertaModel,
    XLNetModel,
)
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

from mosaic.attentional_pooling import AttentionalPooling
from mosaic.custom_coca import prepare_inputs_for_generation
from mosaic.model_builders import (
    build_text_tower,
)
from mosaic.model_configs import TextConfig
from mosaic.utils import _token_to_tensor

try:
    from transformers import (
        BeamSearchScorer,
        EosTokenCriteria,
        LogitsProcessorList,
        MaxLengthCriteria,
        MinLengthLogitsProcessor,
        RepetitionPenaltyLogitsProcessor,
        StoppingCriteriaList,
        TopKLogitsWarper,
        TopPLogitsWarper,
    )

    GENERATION_TYPES = {
        "top_k": TopKLogitsWarper,
        "top_p": TopPLogitsWarper,
        "beam_search": "beam_search",
    }
    _has_transformers = True
except ImportError:
    GENERATION_TYPES = {"top_k": None, "top_p": None, "beam_search": "beam_search"}
    _has_transformers = False


class CustomLM(nn.Module):
    def __init__(self, text_cfg: "TextConfig"):
        """
        Args:
            text_cfg (TextConfig): Configuration for the text encoder.
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

        # Projection layer to vocabulary space
        self.text_projection = nn.Linear(self.config.embed_dim, self.config.vocab_size)
        self.ln_final = LayerNorm(self.config.embed_dim)

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
        input_ids,
        images=None,
        attention_mask: Optional[torch.Tensor] = None,
        image_attention_mask=None,
        output_labels: bool = True,
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        if attention_mask is None:
            attention_mask = torch.ones(
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=torch.float,
                device=input_ids.device,
            )
        # Prepare labels for teacher-forcing if needed
        labels: Optional[torch.Tensor] = input_ids[:, 1:] if output_labels else None
        if output_labels:
            # Align text embeddings and logits with labels for teacher-forcing
            input_ids = input_ids[:, :-1]
            attention_mask = attention_mask[:, 1:]

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

        last_hidden_state = self.ln_final(other_tokens)
        logits = self.text_projection(last_hidden_state)

        return {
            "text_features": text_cls_token,
            "image_features": None,
            "logits": logits,
            "logit_scale": None,
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
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    sot_token_id=sot_token_id,
                    num_beams=num_beams,
                    num_beam_groups=num_beam_groups,
                    min_seq_len=min_seq_len,
                    stopping_criteria=stopping_criteria,
                    logit_processor=logit_processor,
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
                    input_ids=x,
                    output_labels=False,
                )[
                    "logits"
                ][:, -1]
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
    ):
        device = image_inputs.device
        batch_size = image_inputs.shape[0]
        image_inputs = torch.repeat_interleave(image_inputs, num_beams, dim=0)

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
                input_ids=model_inputs["text"],
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
