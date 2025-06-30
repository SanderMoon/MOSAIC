import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer

from mosaic.model_factory import configure_precision
from simplified.pooler_encoder import PoolerOnlyEncoder
from simplified.simple_text_encoder import TextEncoder


class SimpleCLIPModel(nn.Module):
    """
    A simplified CLIP-like model that encodes images and text into a shared embedding space.

    Args:
        vision_encoder (VisionEncoder): The vision encoder module.
        text_encoder (TextEncoder): The text encoder module.
    """

    def __init__(
        self,
        vision_encoder,
        text_encoder,
        init_logit_scale: float = np.log(1 / 0.07),
    ):
        super(SimpleCLIPModel, self).__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder

        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)

    def forward(
        self, texts, images, text_attention_mask, image_features_attention_mask
    ):
        """
        Forward pass for the SimpleCLIPModel.

        Args:
            image_features (torch.Tensor): Image features of shape (B, N, D).
            image_attention_mask (torch.Tensor): Image attention mask of shape (B, N).
            input_ids (torch.Tensor): Tokenized text input IDs of shape (B, L).
            attention_mask (torch.Tensor): Text attention mask of shape (B, L).

        Returns:
            torch.Tensor, torch.Tensor: Normalized image and text embeddings of shape (B, D).
        """
        # Encode images and text
        image_embeddings = self.vision_encoder(
            images, image_features_attention_mask
        )  # Shape: [B, D]
        text_embeddings = self.text_encoder(texts, text_attention_mask)  # Shape: [B, D]

        # Normalize embeddings
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)

        return {
            "text_features": text_embeddings,
            "image_features": image_embeddings,
            "logit_scale": self.logit_scale.exp(),
        }


def create_clip_model(device, precision):
    text_encoder_name = "bert-base-uncased"
    common_dim = 1024
    image_features_dim = 192
    init_log_scale = np.log(1 / 0.07)
    text_encoder = TextEncoder(
        bert_model_name=text_encoder_name, num_heads=8, output_dim=common_dim
    )
    tokenizer = BertTokenizer.from_pretrained(text_encoder_name)
    vision_encoder = PoolerOnlyEncoder(
        input_dim=image_features_dim, num_heads=8, output_dim=common_dim
    )
    model = SimpleCLIPModel(vision_encoder, text_encoder, init_log_scale)
    model, amp, input_dtype = configure_precision(model, precision, device)

    return model, tokenizer, amp, input_dtype
