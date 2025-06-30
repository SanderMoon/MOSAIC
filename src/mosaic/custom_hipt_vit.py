import logging
import math
from functools import partial
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

"""
Modified version of the Vision Transformer (ViT) model for Whole Slide Images (WSIs) based on Lucassen et al.:
https://github.com/RTLucassen/melanocytic_lesion_triaging

Which was based on the modified Pytorch implementation of Vision Transformer (ViT) by by Chen et al.:
https://github.com/mahmoodlab/HIPT

Which was in part copied from the original DINO implementation by Caron et al.:
https://github.com/facebookresearch/dino

Which was in part copied from the timm library by Ross Wightman:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py


"""


def _no_grad_trunc_normal_(
    tensor: Tensor, mean: float, std: float, a: float, b: float
) -> Tensor:
    """
    Fills the input tensor with values drawn from a truncated normal distribution.

    This function is based on the implementation in PyTorch's master branch and will
    be available in future official releases. It generates values from a truncated
    normal distribution using a combination of uniform sampling and the inverse
    cumulative distribution function (CDF) for the normal distribution.

    Args:
        tensor (Tensor): The tensor to fill with values from the truncated normal distribution.
        mean (float): The mean of the truncated normal distribution.
        std (float): The standard deviation of the truncated normal distribution.
        a (float): The lower bound of the truncation interval.
        b (float): The upper bound of the truncation interval.

    Returns:
        Tensor: The input tensor filled with values from the truncated normal distribution.
    """

    def norm_cdf(x: float) -> float:
        """
        Computes the standard normal cumulative distribution function (CDF).

        Args:
            x (float): The input value.

        Returns:
            float: The CDF value for the given input.
        """
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        logger.warning(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        lower = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        tensor.uniform_(2 * lower - 1, 2 * u - 1)

        tensor.erfinv_()

        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(
    tensor: Tensor, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0
) -> Tensor:
    """
    Fills the input tensor with values drawn from a truncated normal distribution.

    This is a convenience wrapper around `_no_grad_trunc_normal_` that provides
    default values for the mean, standard deviation, and truncation bounds.

    Args:
        tensor (Tensor): The tensor to fill.
        mean (float, optional): The mean of the truncated normal distribution. Defaults to 0.0.
        std (float, optional): The standard deviation of the truncated normal distribution.
            Defaults to 1.0.
        a (float, optional): The lower bound of the truncation interval. Defaults to -2.0.
        b (float, optional): The upper bound of the truncation interval. Defaults to 2.0.

    Returns:
        Tensor: The input tensor filled with values from the truncated normal distribution.
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class CustomTransformerEncoderLayer(nn.Module):
    """
    A custom Transformer Encoder Layer

    Args:
        d_model: The number of expected features in the input.
        nhead: The number of heads in the multihead attention models.
        dim_feedforward: The dimension of the feedforward network model.
        dropout_attn: Dropout rate specifically for attention weights.
        dropout_ffn: Dropout rate specifically for the feedforward network.
        dropout_residual: Dropout rate for the residual connections.
        activation: The activation function of the intermediate layer, can be a string
                    ("relu" or "gelu") or a unary callable. Default: "gelu".
        layer_norm_eps: The eps value in layer normalization components.
        batch_first: If `True`, then the input and output tensors are provided
                     as (batch, seq, feature). Default: `False`.
        norm_first: If `True`, layer norm is done prior to attention and feedforward
                   operations, respectively. Otherwise, it's done after.
        bias: If set to `False`, Linear and LayerNorm layers will not learn an additive
              bias. Default: `True`.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout_attn: float = 0.0,
        dropout_ffn: float = 0.0,
        dropout_residual: float = 0.0,
        activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = "gelu",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
    ) -> None:
        super(CustomTransformerEncoderLayer, self).__init__()
        if d_model % nhead != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by nhead ({nhead})"
            )

        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout_attn, batch_first=batch_first, bias=bias
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias)
        self.dropout_ffn = nn.Dropout(dropout_ffn)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.dropout_residual1 = nn.Dropout(dropout_residual)
        self.dropout_residual2 = nn.Dropout(dropout_residual)

        if isinstance(activation, str):
            activation = self._get_activation_fn(activation)
        elif not callable(activation):
            raise ValueError("Activation must be a string or a callable.")
        self.activation = activation

    def _get_activation_fn(
        self, activation: str
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu
        else:
            raise RuntimeError(f"Unsupported activation: {activation}")

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            src: (batch_size, seq_length, d_model) if batch_first=True,
                 otherwise (seq_length, batch_size, d_model)
        """
        if self.norm_first:
            src2 = self.norm1(src)
            attn_output, _ = self.self_attn(
                src2,
                src2,
                src2,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
            )
            src = src + self.dropout_residual1(attn_output)
            src2 = self.norm2(src)
            ffn_output = self.linear2(
                self.dropout_ffn(self.activation(self.linear1(src2)))
            )
            src = src + self.dropout_residual2(ffn_output)
        else:
            attn_output, _ = self.self_attn(
                src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
            )
            src = src + self.dropout_residual1(attn_output)
            src = self.norm1(src)
            ffn_output = self.linear2(
                self.dropout_ffn(self.activation(self.linear1(src)))
            )
            src = src + self.dropout_residual2(ffn_output)
            src = self.norm2(src)
        return src


class VisionTransformerWSI(nn.Module):
    """
    Vision Transformer adapted for Whole Slide Images (WSIs).

    (Same docstring as before)
    """

    def __init__(
        self,
        input_embed_dim: int = 384,
        output_embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout_attn: float = 0.0,
        dropout_ffn: float = 0.0,
        dropout_residual: float = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        add_cls_token: bool = False,
    ):
        super().__init__()
        embed_dim = output_embed_dim
        self.num_features = self.embed_dim = embed_dim

        self.phi = nn.Sequential(
            nn.Linear(input_embed_dim, output_embed_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_ffn),
        )

        self.pos_drop = nn.Dropout(p=dropout_ffn)

        self.num_heads = num_heads

        self.blocks = nn.ModuleList(
            [
                CustomTransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=int(embed_dim * mlp_ratio),
                    dropout_attn=dropout_attn,
                    dropout_ffn=dropout_ffn,
                    dropout_residual=dropout_residual,
                    activation="gelu",
                    layer_norm_eps=1e-5,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(depth)
            ]
        )

        self.norm = norm_layer(embed_dim)
        self.add_cls_token = add_cls_token

        if self.add_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self._init_weights()

    def _init_weights(self):
        """
        Initializes the model's weights using truncated normal initialization.
        """
        if self.add_cls_token:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights_recursive)

    def _init_weights_recursive(self, m: nn.Module):
        """
        Recursively initializes weights for linear and layer normalization layers.
        """
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def prepare_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        Prepares input tokens by flattening, applying linear projection, and
        optionally adding a CLS token

        Args:
            x (Tensor): Input tensor of shape (batch_size, embed_dim, width, height)

        Returns:
            Tensor: Prepared tokens with shape (batch_size, seq_len, embed_dim)
        """
        B, N, embed_dim = x.shape
        x = self.phi(x)
        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        return self.pos_drop(x)

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, N, embed_dim = x.shape

        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = ~attention_mask.bool()

        x = self.prepare_tokens(x)
        for block in self.blocks:
            x = block(x, src_mask=None, src_key_padding_mask=src_key_padding_mask)
        x = self.norm(x)
        if self.add_cls_token:
            return x[:, 1:], x[:, 0]
        else:
            return x

    def get_last_selfattention(self, x: Tensor) -> Tensor:
        """
        Retrieves the attention weights from the last self-attention layer.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Attention weights from the last self-attention layer.

        Raises:
            ValueError: If `self.blocks` is empty.
        """
        x = self.prepare_tokens(x)

        if not self.blocks:
            raise ValueError(
                "Cannot get attention from the last block because self.blocks is empty."
            )

        for i, block in enumerate(self.blocks):
            x = block(x)
            if i == len(self.blocks) - 1:
                break

        last_block = self.blocks[-1]

        _, attn_weights = last_block.self_attn(
            query=x, key=x, value=x, need_weights=True
        )

        return attn_weights

    def get_intermediate_layers(self, x: Tensor, n: int = 1) -> List[Tensor]:
        """
        Retrieves intermediate layer outputs from the Vision Transformer.

        Args:
            x (Tensor): Input tensor.
            n (int, optional): Number of intermediate layers to return. Defaults to 1.

        Returns:
            List[Tensor]: A list of intermediate layer outputs.
        """
        x = self.prepare_tokens(x)
        output = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


def vit_wsi_4k_paper(**kwargs) -> VisionTransformerWSI:
    """
    Creates a Vision Transformer WSI model with specific configurations
    as described in the paper.

    Args:
        **kwargs: Additional keyword arguments to pass to the `VisionTransformerWSI` constructor

    Returns:
        VisionTransformerWSI: The initialized Vision Transformer WSI model
    """
    model = VisionTransformerWSI(
        input_embed_dim=192,
        output_embed_dim=1024,
        depth=2,
        num_heads=4,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model
