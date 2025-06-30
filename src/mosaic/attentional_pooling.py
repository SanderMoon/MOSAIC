import torch
from torch import nn


class AttentionalPooling(nn.Module):
    """
    This class implements an attentional pooling strategy for reducing output tokens to the
    size of a learnable query matrix or vector using cross-attention.

    This is often referred to as the "Perceiver Resampler" or "Attentional Pooling" in the context of vision transformers and other models.

    Args:
        dim (int): The embedding dimension of the input features and query.
        num_heads (int, optional): The number of attention heads. Defaults to 8.
    """

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.attention = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(dim)

    def forward(
        self, features: torch.Tensor, query: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies attentional pooling to the input features using the given query.

        Args:
            features (torch.Tensor): The input features of shape (B, N, D).
            query (torch.Tensor): The query vector of shape (B, R, D).
            attention_mask (torch.Tensor): The attention mask of shape (B, N),
                                           where `1` indicates valid positions and `0` indicates masked positions.

        Returns:
            torch.Tensor: The pooled output of shape (B, R, D).
        """

        B, N, D = features.size()
        B_q, R, D_q = query.size()

        # Validate dimensions
        if D != self.dim:
            raise ValueError(
                f"Features embedding dimension ({D}) does not match expected dimension ({self.dim})."
            )
        if D_q != self.dim:
            raise ValueError(
                f"Query embedding dimension ({D_q}) does not match expected dimension ({self.dim})."
            )

        # Check batch sizes
        if B_q == 1 and B > 1:
            # Shared query across the batch, expand it
            query_expanded = query.repeat(B, 1, 1)  # (B, R, D)
        elif B_q == B:
            # Batch-specific query
            query_expanded = query
        else:
            raise ValueError(
                f"Query batch size ({B_q}) does not match features batch size ({B}), and is not 1."
            )

        # Prepare the key_padding_mask
        # In PyTorch, `key_padding_mask` expects `True` for positions to be masked
        # and `False` for positions to be kept. If your `attention_mask` has `1` for
        # valid positions and `0` for masked, invert it.
        key_padding_mask = ~attention_mask.bool()  # Shape: (B, N)

        # Apply multi-head attention with key_padding_mask
        attn_output, _ = self.attention(
            query_expanded, features, features, key_padding_mask=key_padding_mask
        )

        # Apply layer normalization
        out = self.layer_norm(attn_output)

        return out
