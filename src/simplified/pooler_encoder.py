from dataclasses import dataclass

from torch import nn


@dataclass
class PoolerOnlyConfig:
    pooler_num_heads: int = 8


class PoolerOnlyEncoder(nn.Module):
    """
    Vision Encoder that processes pre-embedded image features and produces a single embedding.

    Args:
        feature_dim (int): The dimensionality of the input image features.
        num_heads (int, optional): Number of attention heads in AttentionalPooling. Defaults to 8.
    """

    def __init__(self, input_dim, num_heads=8, output_dim=1024):
        super(PoolerOnlyEncoder, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads

        # upsample the input features to the common dim
        self.phi = nn.Sequential(nn.Linear(input_dim, output_dim), nn.GELU())

    def forward(self, image_features, image_attention_mask):
        """
        Forward pass for the VisionEncoder.

        Args:
            image_features (torch.Tensor): Image features of shape (B, N, D).
            image_attention_mask (torch.Tensor): Attention mask of shape (B, N), where 1 indicates valid positions.

        Returns:
            torch.Tensor: Pooled image embeddings of shape (B, D).
        """
        B, N, D = image_features.size()

        # upsample the input features to the common dim
        image_features = self.phi(image_features)

        return image_features
