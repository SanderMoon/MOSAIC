"""
This is a PyTorch implementation of the Perceiver Encoder, inspired by the original from Google DeepMind:
https://github.com/google-deepmind/deepmind-research/blob/master/perceiver/perceiver.@property

The code includes additional key-value caching, and also includes the implementation of the PRISM Perceiver variant explained in the papep by Shaikovski et al. (2024):
https://arxiv.org/abs/2405.10254

However, this implementation does not use any code from the PRISM project directly.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def attend(q, k, v, dropout_prob=0.0, attention_mask=None):
    """Computes multi-head attention using a query, key and value.

    Args:
        q: Query with shape [batch, q_indices, num_heads, head_dim].
        k: Key with shape [batch, kv_indices, num_heads, head_dim].
        v: Value with shape [batch, kv_indices, num_heads, head_dim].
        dropout_prob: dropout probability on the attention weights.
        attention_mask: Array of shape [batch, q_indices, kv_indices] indicating
        which attentions are valid
    Returns:
        Output of the attention with shape [batch, q_indices, hiddens]
    """
    batch, q_indices, num_heads, q_head_dim = q.shape
    _, _, _, v_head_dim = v.shape
    hiddens = num_heads * v_head_dim

    # Compute attention scores using einsum: 'bthd,bThd->bhtT'
    # NOTE to self: The letters b, t, T, h, and d represent specific dimensions:
    #   b (batch), t (target sequence length), T (source sequence length), h (num_heads), and d (head_dim).
    # This einsum performs a batched matrix multiplication of q and k across each head for each sequence position:
    #   It takes the dot product over the last dimension d (head dimension), creating an output of shape [batch_size, num_heads, target_seq_len, source_seq_len]

    attention = torch.einsum("bthd,bThd->bhtT", q, k)

    # Scale attention scores
    scale = 1.0 / math.sqrt(q_head_dim)
    attention = attention * scale

    if attention_mask is not None:
        large_k = 1e4 if attention.dtype == torch.float16 else 1e30
        # Shape: [1, 1, 1, 1] to ensure proper broadcasting
        large_k = torch.tensor(large_k, device=attention.device, dtype=attention.dtype)
        attention_mask = attention_mask.unsqueeze(1).bool()
        attention = torch.where(attention_mask, attention, -large_k)

    # Apply softmax to get normalized attention weights
    normalized = F.softmax(attention, dim=-1)

    if dropout_prob > 0.0:
        normalized = F.dropout(normalized, p=dropout_prob)

    # Compute the weighted sum of values using einsum: 'bhtT,bThd->bthd'
    summed = torch.einsum("bhtT,bThd->bthd", normalized, v)

    # Reshape to [batch, q_indices, hiddens]
    summed = summed.contiguous().view(batch, q_indices, hiddens)

    if attention_mask is not None:
        # If all attended tokens are masked, force the output to zero
        wipe_attn = (~attention_mask).all(
            dim=-1, keepdim=True
        )  # [batch, 1, q_indices, 1]
        # Remove singleton dimensions to get [batch, q_indices]
        wipe_attn = wipe_attn.squeeze(1)

        # Apply the wipe_attn mask
        summed = torch.where(wipe_attn, torch.zeros_like(summed), summed)

    return summed


def conv_1d(output_channels, in_features, init_scale=1.0, with_bias=True):
    """
    A 1D convolution implemented as a linear layer.

    Args:
        output_channels: Number of output channels.
        in_features: Number of input features.
        init_scale: Scaling factor for weight initialization.
        with_bias: Whether to include a bias term.

    Returns:
        A PyTorch nn.Linear layer initialized with Kaiming uniform scaling.
    """
    linear = nn.Linear(
        in_features=in_features, out_features=output_channels, bias=with_bias
    )
    # Initialize weights with Kaiming uniform to match VarianceScaling(init_scale, mode='fan_in', nonlinearity='linear')
    nn.init.kaiming_uniform_(linear.weight, a=0, mode="fan_in", nonlinearity="linear")

    if with_bias:
        nn.init.zeros_(linear.bias)

    return linear


def layer_norm(normalized_shape, eps=1e-5, elementwise_affine=True):
    """
    Applies Layer Normalization as a PyTorch module.

    Args:
        normalized_shape: Input shape from an expected input of size
                          [*, normalized_shape[0], normalized_shape[1], ...].
        eps: A value added to the denominator for numerical stability.
        elementwise_affine: A boolean value that when set to True,
                             this module has learnable affine parameters.

    Returns:
        A PyTorch nn.LayerNorm module.
    """
    return nn.LayerNorm(
        normalized_shape=normalized_shape,
        eps=eps,
        elementwise_affine=elementwise_affine,
    )


def make_cross_attention_mask(query_mask, kv_mask):
    """
    Creates a cross-attention mask by computing the outer product of query and key-value masks.

    Args:
        query_mask: Tensor of shape [batch_size, query_len].
        kv_mask: Tensor of shape [batch_size, key_len].

    Returns:
        mask: Tensor of shape [batch_size, query_len, key_len] indicating valid attentions.
    """
    # Compute outer product for each batch
    mask = query_mask.unsqueeze(-1) * kv_mask.unsqueeze(1)
    return mask


#########################
#       Modules         #
#########################

"""
Follows:
https://github.com/google-deepmind/deepmind-research/blob/master/perceiver/perceiver.py

Changes:
- Added qk_channels and v_channels to the constructor, given that in pytorch it's best to initialize layers in the init function
- Added _initialize_layers function to initialize the layers based on the input dimensions
"""


class Attention(nn.Module):
    """Multi-headed {cross, self}-attention."""

    def __init__(
        self,
        input_dim_q,
        input_dim_kv=None,
        num_heads=8,
        init_scale=1.0,
        with_final_bias=True,
        final_init_scale_multiplier=1.0,
        dropout_prob=0.0,
        output_channels=None,
        qk_channels=None,
        v_channels=None,
    ):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.init_scale = init_scale
        self.with_final_bias = with_final_bias
        self.final_init_scale = final_init_scale_multiplier * init_scale
        self.dropout_prob = dropout_prob
        self.qk_channels = qk_channels
        self.v_channels = v_channels
        self.output_channels = output_channels

        self.input_dim_q = input_dim_q
        # Default to self-attention if input_dim_kv is not provided
        if input_dim_kv is None:
            self.input_dim_kv = input_dim_q
        else:
            self.input_dim_kv = input_dim_kv

        self._initialize_layers()

    def _initialize_layers(self):
        """Initialize Q, K, V projections and final output projection based on input dimensions."""
        if self.qk_channels is None:
            self.qk_channels = self.input_dim_q
        if self.v_channels is None:
            self.v_channels = self.qk_channels
        if self.output_channels is None:
            self.output_channels = self.v_channels

        if self.qk_channels % self.num_heads != 0:
            raise ValueError(
                f"qk_channels ({self.qk_channels}) must be divisible by num_heads ({self.num_heads})."
            )
        if self.v_channels % self.num_heads != 0:
            raise ValueError(
                f"v_channels ({self.v_channels}) must be divisible by num_heads ({self.num_heads})."
            )

        # Define projection layers using conv_1d
        self.q_proj = conv_1d(
            self.qk_channels,
            self.input_dim_q,
            init_scale=self.init_scale,
            with_bias=False,
        )
        self.k_proj = conv_1d(
            self.qk_channels,
            self.input_dim_kv,
            init_scale=self.init_scale,
            with_bias=False,
        )
        self.v_proj = conv_1d(
            self.v_channels,
            self.input_dim_kv,
            init_scale=self.init_scale,
            with_bias=False,
        )
        self.out_proj = conv_1d(
            self.output_channels,
            self.v_channels,
            init_scale=self.final_init_scale,
            with_bias=self.with_final_bias,
        )

    def forward(self, inputs_q, inputs_kv=None, kv_cache=None, attention_mask=None):
        """
        Args:
            inputs_q: Tensor of shape [batch_size, query_len, input_dim_q].
            inputs_kv: Tensor of shape [batch_size, key_len, input_dim_kv].
            kv_cache: Tuple of (key_cache, value_cache), each with shape
                      [batch_size, past_key_len, num_heads, head_dim].
            attention_mask: Optional tensor of shape [batch_size, query_len, key_len].

        Returns:
            Tuple:
                - Output tensor of shape [batch_size, query_len, output_channels].
                - Updated kv_cache (tuple of key_cache and value_cache).
        """
        batch_size, q_len, _ = inputs_q.shape
        qk_channels_per_head = self.qk_channels // self.num_heads
        v_channels_per_head = self.v_channels // self.num_heads

        # Project queries
        q = self.q_proj(inputs_q).view(
            batch_size, q_len, self.num_heads, qk_channels_per_head
        )

        if kv_cache is None:
            # Project keys and values for the first time
            assert (
                inputs_kv is not None
            ), "inputs_kv must be provided if kv_cache is None"
            k = self.k_proj(inputs_kv).view(
                batch_size, -1, self.num_heads, qk_channels_per_head
            )
            v = self.v_proj(inputs_kv).view(
                batch_size, -1, self.num_heads, v_channels_per_head
            )
        else:
            # Use cached keys and values
            k_cache, v_cache = kv_cache

            if inputs_kv is not None:
                # Extend the cache with new keys and values
                k_new = self.k_proj(inputs_kv).view(
                    batch_size, -1, self.num_heads, qk_channels_per_head
                )
                v_new = self.v_proj(inputs_kv).view(
                    batch_size, -1, self.num_heads, v_channels_per_head
                )
                k = torch.cat([k_cache, k_new], dim=1)
                v = torch.cat([v_cache, v_new], dim=1)
            else:
                k = k_cache
                v = v_cache

        # Apply attention
        result = attend(
            q, k, v, dropout_prob=self.dropout_prob, attention_mask=attention_mask
        )

        # Final projection
        output = self.out_proj(result.view(batch_size, q_len, self.output_channels))

        # Update the cache
        new_kv_cache = (k, v)

        return output, new_kv_cache


"""
Main changes:
- Initialize the dense layers in the init function
"""


class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)


class MLP(nn.Module):
    """A Transformer-style dense module to follow attention."""

    def __init__(
        self,
        input_dim,
        widening_factor=4,
        dropout_prob=0.0,
        init_scale=1.0,
        activation="gelu",
    ):
        """
        Initializes the MLP module.

        Args:
            input_dim (int): Dimension of the input features.
            widening_factor (int, optional): Factor to widen the hidden layer. Default is 4.
            dropout_prob (float, optional): Dropout probability. Default is 0.0.
            init_scale (float, optional): Scale factor for weight initialization. Default is 1.0.
        """
        super(MLP, self).__init__()
        self.dropout_prob = dropout_prob

        # Set widening factor to 2x for GEGLU activation to accommodate chunking
        if activation == "geglu":
            widening_factor *= 2

        # Initialize the first convolutional layer
        self.dense1 = conv_1d(
            in_features=input_dim,
            output_channels=widening_factor * input_dim,
            init_scale=init_scale,
        )

        # Initialize the second convolutional layer
        self.dense2 = conv_1d(
            in_features=widening_factor
            * input_dim
            // (2 if activation == "geglu" else 1),
            output_channels=input_dim,
            init_scale=init_scale,
        )

        # Initialize the activation function
        self.activation = self.get_activation_function(activation)

    def get_activation_function(self, activation):
        if isinstance(activation, str):
            activation = activation.lower()
            if activation == "relu":
                return F.relu
            elif activation == "gelu":
                return F.gelu
            elif activation == "sigmoid":
                return torch.sigmoid
            elif activation == "tanh":
                return torch.tanh
            elif activation == "geglu":
                return GEGLU()
            else:
                raise ValueError(f"Unsupported activation function: {activation}")
        elif callable(activation):
            return activation
        else:
            raise TypeError("Activation must be either a string or a callable function")

    def forward(self, x, *, is_training):
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dense2(x)
        return F.dropout(x, p=self.dropout_prob, training=is_training)


"""
Main changes:
- Initialize the layers in the init function
"""


class SelfAttention(nn.Module):
    """A self-attention module, including a dense block."""

    def __init__(
        self,
        input_dim,
        widening_factor=4,
        dropout_prob=0.0,
        dropout_attn_prob=0.0,
        num_heads=8,
        att_init_scale=1.0,
        dense_init_scale=1.0,
        qk_channels=None,
        v_channels=None,
        mlp_activation="gelu",
    ):
        super(SelfAttention, self).__init__()
        self._input_dim = input_dim
        self._widening_factor = widening_factor
        self._dropout_prob = dropout_prob
        self._dropout_attn_prob = dropout_attn_prob
        self._num_heads = num_heads
        self._att_init_scale = att_init_scale
        self._dense_init_scale = dense_init_scale
        self._qk_channels = qk_channels
        self._v_channels = v_channels
        self._mlp_activation = mlp_activation

        self._initialize_layers()

    def _initialize_layers(self):
        """Initialize the attention and MLP layers based on the input dimensions."""
        self.attention = Attention(
            input_dim_q=self._input_dim,
            input_dim_kv=self._input_dim,
            num_heads=self._num_heads,
            init_scale=self._att_init_scale,
            qk_channels=self._qk_channels,
            v_channels=self._v_channels,
            dropout_prob=self._dropout_attn_prob,
        )
        self.mlp = MLP(
            input_dim=self.attention.v_channels,  # v vector decides output size of attention
            widening_factor=self._widening_factor,
            dropout_prob=self._dropout_prob,
            init_scale=self._dense_init_scale,
            activation=self._mlp_activation,
        )

        self.ln1 = nn.LayerNorm(self.attention.v_channels)
        self.ln2 = nn.LayerNorm(self.attention.v_channels)

    def forward(self, inputs, *, attention_mask=None, is_training):
        x = inputs
        qkv_inputs = self.ln1(inputs)
        attention, _ = self.attention(
            qkv_inputs, qkv_inputs, attention_mask=attention_mask
        )
        attention = F.dropout(attention, p=self._dropout_prob, training=is_training)
        x = torch.add(x, attention)
        mlp_output = self.mlp(self.ln2(x), is_training=is_training)
        x = torch.add(x, mlp_output)
        return x


class CrossAttention(nn.Module):
    """A cross-attention module, including a dense block."""

    def __init__(
        self,
        input_dim_q,
        input_dim_kv,
        qk_channels,
        v_channels,
        widening_factor=1,
        dropout_prob=0.0,
        dropout_attn_prob=0.0,
        num_heads=8,
        att_init_scale=1.0,
        dense_init_scale=1.0,
        use_query_residual=True,
        mlp_activation="gelu",
    ):
        super(CrossAttention, self).__init__()
        self.input_dim_q = input_dim_q
        self.input_dim_kv = input_dim_kv
        self._widening_factor = widening_factor
        self._dropout_prob = dropout_prob
        self._dropout_attn_prob = dropout_attn_prob
        self._num_heads = num_heads
        self._att_init_scale = att_init_scale
        self._dense_init_scale = dense_init_scale
        self._use_query_residual = use_query_residual
        self._qk_channels = qk_channels
        self._v_channels = v_channels
        self._mlp_activation = mlp_activation

        self._initialize_layers()

    def _initialize_layers(self):
        """Initialize the attention and MLP layers based on the input dimensions."""
        self.attention = Attention(
            input_dim_q=self.input_dim_q,
            input_dim_kv=self.input_dim_kv,
            num_heads=self._num_heads,
            init_scale=self._att_init_scale,
            qk_channels=self._qk_channels,
            v_channels=self._v_channels,
            dropout_prob=self._dropout_attn_prob,
        )
        self.mlp = MLP(
            input_dim=self.attention.v_channels,  # v vector decides output size of attention
            widening_factor=self._widening_factor,
            dropout_prob=self._dropout_prob,
            init_scale=self._dense_init_scale,
            activation=self._mlp_activation,
        )

        self.ln1 = layer_norm(self.input_dim_q)
        self.ln2 = layer_norm(self.input_dim_kv)
        self.ln3 = layer_norm(self.attention.v_channels)

    def forward(
        self, inputs_q, inputs_kv, *, attention_mask=None, kv_cache=None, is_training
    ):
        q = self.ln1(inputs_q)

        if kv_cache is None:
            kv = self.ln2(inputs_kv)
        else:
            kv = None

        attention, new_kv_cache = self.attention(
            q, kv, kv_cache=kv_cache, attention_mask=attention_mask
        )
        attention = F.dropout(
            attention, p=self._dropout_attn_prob, training=is_training
        )

        if self._use_query_residual:
            x = torch.add(inputs_q, attention)
        else:
            x = attention

        mlp_output = self.mlp(self.ln3(x), is_training=is_training)
        x = torch.add(x, mlp_output)
        return x, new_kv_cache


class PerceiverEncoder(nn.Module):
    """
    The Perceiver Encoder: a scalable, fully attentional encoder.

    Constructs a Perceiver-like encoder with cross-attention and multiple self-attention layers.
    The latent array `z` is initialized as a set of trainable embeddings.

    Args:
        num_self_attends_per_block (int): Number of self-attention layers per block.
        num_blocks (int): Number of blocks to stack.
        z_index_dim (int): Number of latent embeddings.
        num_z_channels (int): Dimensionality of each latent embedding.
        qk_channels (int, optional): Dimensionality for query and key projections. Defaults to None.
        v_channels (int, optional): Dimensionality for value projections. Defaults to None.
        num_cross_attend_heads (int): Number of heads in cross-attention.
        num_self_attend_heads (int): Number of heads in self-attention.
        cross_attend_widening_factor (int): Widening factor for cross-attention MLP.
        self_attend_widening_factor (int): Widening factor for self-attention MLP.
        dropout_prob (float): Dropout probability for MLP layers.
        z_pos_enc_init_scale (float): Initialization scale for latent embeddings.
        cross_attention_shape_for_attn (str): Shape configuration for attention (unused in this implementation).
        use_query_residual (bool): Whether to use residual connections for queries in cross-attention.
        name (str): Name of the module (unused in PyTorch but kept for consistency).
    """

    def __init__(
        self,
        input_dim,
        output_dim=None,
        num_self_attends_per_block=6,
        num_blocks=8,
        z_index_dim=512,
        num_z_channels=1024,
        qk_channels=None,
        num_cross_attend_heads=1,
        num_self_attend_heads=8,
        cross_attend_widening_factor=1,
        self_attend_widening_factor=1,
        dropout_prob=0.0,
        dropout_attn_prob=0.0,
        z_pos_enc_init_scale=0.02,
        att_init_scale=1.0,
        dense_init_scale=1.0,
        use_query_residual=True,  # Cannot be True if v_channels != num_z_channels
        mlp_activation="gelu",
        name="perceiver_encoder",  # Not used in PyTorch
    ):
        super(PerceiverEncoder, self).__init__()
        self.num_blocks = num_blocks
        self.num_self_attends_per_block = num_self_attends_per_block
        self.num_latents = z_index_dim
        self.num_z_channels = num_z_channels

        # Initialize the latent array `z` as trainable embeddings
        self.z = nn.Parameter(
            torch.randn(self.num_latents, self.num_z_channels) * z_pos_enc_init_scale
        )

        # Initialize Cross Attention
        self.cross_attend = CrossAttention(
            input_dim_q=num_z_channels,
            input_dim_kv=input_dim,
            qk_channels=qk_channels,
            v_channels=num_z_channels,
            widening_factor=cross_attend_widening_factor,
            dropout_prob=dropout_prob,
            dropout_attn_prob=dropout_attn_prob,  # Assuming same dropout for attention
            num_heads=num_cross_attend_heads,
            att_init_scale=att_init_scale,  # Default scale
            dense_init_scale=dense_init_scale,  # Default scale
            use_query_residual=use_query_residual,
            mlp_activation=mlp_activation,
        )

        # Initialize Self Attention layers
        self.self_attends = nn.ModuleList(
            [
                SelfAttention(
                    input_dim=num_z_channels,
                    qk_channels=qk_channels,
                    v_channels=num_z_channels,
                    widening_factor=self_attend_widening_factor,
                    dropout_prob=dropout_prob,
                    dropout_attn_prob=dropout_attn_prob,  # Assuming same dropout for attention
                    num_heads=num_self_attend_heads,
                    att_init_scale=att_init_scale,  # Default scale
                    dense_init_scale=dense_init_scale,  # Default scale
                    mlp_activation=mlp_activation,
                )
                for i in range(num_self_attends_per_block)
            ]
        )

        if output_dim != num_z_channels:
            self.output_projection = conv_1d(
                in_features=num_z_channels, output_channels=output_dim, init_scale=1.0
            )

    def forward(
        self, inputs: torch.Tensor, input_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass of the Perceiver Encoder.

        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_dim).
            input_mask (torch.Tensor, optional): Mask tensor of shape (batch_size, seq_length) indicating valid inputs.

        Returns:
            torch.Tensor: Encoded latents of shape (batch_size, num_latents, num_z_channels).
        """
        batch_size = inputs.size(0)

        # Expand the latent embeddings `z` to match the batch size
        # Shape: (batch_size, num_latents, num_z_channels)
        z = self.z.unsqueeze(0).expand(batch_size, -1, -1)

        # Create attention mask if input_mask is provided
        if input_mask is not None:
            # Assuming input_mask is of shape (batch_size, seq_length)
            # and that queries (z) are all valid (no masking needed)
            query_mask = torch.ones(
                (batch_size, self.num_latents), dtype=torch.bool, device=inputs.device
            )
            kv_mask = input_mask.bool()
            attention_mask = make_cross_attention_mask(query_mask, kv_mask)
        else:
            attention_mask = None

        # Apply Self Attention layers across all blocks
        for _ in range(self.num_blocks):
            # Apply Cross Attention
            z, _ = self.cross_attend(
                inputs_q=z,
                inputs_kv=inputs,
                attention_mask=attention_mask,
                is_training=self.training,
                kv_cache=None,
            )
            for self_attend in self.self_attends:
                z = self_attend(
                    inputs=z,
                    attention_mask=None,  # Assuming self-attention doesn't require a mask
                    is_training=self.training,
                )

        if hasattr(self, "output_projection"):
            z = self.output_projection(z)

        return z


class PrismPerceiverEncoder(nn.Module):
    """
    The Perceiver Encoder: a scalable, fully attentional encoder.

    Constructs a Perceiver-like encoder based on the visual encoder in "PRISM: A MULTI-MODAL GENERATIVE
    FOUNDATION MODEL FOR SLIDE-LEVEL HISTOPATHOLOGY"
    with cross-attention and multiple self-attention layers.
    The latent array `z` is initialized as a set of trainable embeddings.

    Args:
        num_self_attends_per_block (int): Number of self-attention layers per block.
        num_blocks (int): Number of blocks to stack.
        z_index_dim (int): Number of latent embeddings.
        num_z_channels (int): Dimensionality of each latent embedding.
        qk_channels (int, optional): Dimensionality for query and key projections. Defaults to None.
        v_channels (int, optional): Dimensionality for value projections. Defaults to None.
        num_cross_attend_heads (int): Number of heads in cross-attention.
        num_self_attend_heads (int): Number of heads in self-attention.
        cross_attend_widening_factor (int): Widening factor for cross-attention MLP.
        self_attend_widening_factor (int): Widening factor for self-attention MLP.
        dropout_prob (float): Dropout probability for MLP layers.
        z_pos_enc_init_scale (float): Initialization scale for latent embeddings.
        cross_attention_shape_for_attn (str): Shape configuration for attention (unused in this implementation).
        use_query_residual (bool): Whether to use residual connections for queries in cross-attention.
        name (str): Name of the module (unused in PyTorch but kept for consistency).
    """

    def __init__(
        self,
        input_dim,
        output_dim=None,
        num_self_attends_per_block=6,
        num_blocks=8,
        z_index_dim=512,
        num_z_channels=1024,
        qk_channels=None,
        num_cross_attend_heads=1,
        num_self_attend_heads=8,
        cross_attend_widening_factor=1,
        self_attend_widening_factor=1,
        dropout_prob=0.0,
        dropout_attn_prob=0.0,
        z_pos_enc_init_scale=0.02,
        att_init_scale=1.0,
        dense_init_scale=1.0,
        use_query_residual=True,  # Cannot be True if v_channels != num_z_channels
        mlp_activation="gelu",
        name="perceiver_encoder",  # Not used in PyTorch
    ):
        super(PrismPerceiverEncoder, self).__init__()
        self.num_blocks = num_blocks
        self.num_self_attends_per_block = num_self_attends_per_block
        self.num_latents = z_index_dim
        self.num_z_channels = num_z_channels

        # Initialize the latent array `z` as trainable embeddings
        self.z = nn.Parameter(
            torch.randn(self.num_latents, self.num_z_channels) * z_pos_enc_init_scale
        )

        # Initialize Cross Attention
        self.cross_attend_0 = CrossAttention(
            input_dim_q=num_z_channels,
            input_dim_kv=input_dim,
            qk_channels=qk_channels,
            v_channels=num_z_channels,
            widening_factor=cross_attend_widening_factor,
            dropout_prob=dropout_prob,
            dropout_attn_prob=dropout_attn_prob,  # Assuming same dropout for attention
            num_heads=num_cross_attend_heads,
            att_init_scale=att_init_scale,  # Default scale
            dense_init_scale=dense_init_scale,  # Default scale
            use_query_residual=use_query_residual,
            mlp_activation=mlp_activation,
        )

        self.cross_attend_1 = CrossAttention(
            input_dim_q=num_z_channels,
            input_dim_kv=input_dim,
            qk_channels=qk_channels,
            v_channels=num_z_channels,
            widening_factor=cross_attend_widening_factor,
            dropout_prob=dropout_prob,
            dropout_attn_prob=dropout_attn_prob,  # Assuming same dropout for attention
            num_heads=num_cross_attend_heads,
            att_init_scale=att_init_scale,  # Default scale
            dense_init_scale=dense_init_scale,  # Default scale
            use_query_residual=use_query_residual,
            mlp_activation=mlp_activation,
        )

        # Initialize Self Attention layers
        self.self_attends = nn.ModuleList(
            [
                SelfAttention(
                    input_dim=num_z_channels,
                    qk_channels=qk_channels,
                    v_channels=num_z_channels,
                    widening_factor=self_attend_widening_factor,
                    dropout_prob=dropout_prob,
                    dropout_attn_prob=dropout_attn_prob,  # Assuming same dropout for attention
                    num_heads=num_self_attend_heads,
                    att_init_scale=1.0,  # Default scale
                    dense_init_scale=1.0,  # Default scale
                )
                for i in range(num_self_attends_per_block)
            ]
        )

        if output_dim != num_z_channels:
            self.output_projection = conv_1d(
                in_features=num_z_channels, output_channels=output_dim, init_scale=1.0
            )

    def forward(self, inputs: torch.Tensor, input_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Perceiver Encoder.

        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_dim).
            input_mask (torch.Tensor, optional): Mask tensor of shape (batch_size, seq_length) indicating valid inputs.

        Returns:
            torch.Tensor: Encoded latents of shape (batch_size, num_latents, num_z_channels).
        """
        batch_size = inputs.size(0)

        # Expand the latent embeddings `z` to match the batch size
        # Shape: (batch_size, num_latents, num_z_channels)
        z = self.z.unsqueeze(0).expand(batch_size, -1, -1)

        # Create attention mask if input_mask is provided
        if input_mask is not None:
            # Assuming input_mask is of shape (batch_size, seq_length)
            # and that queries (z) are all valid (no masking needed)
            query_mask = torch.ones(
                (batch_size, self.num_latents), dtype=torch.bool, device=inputs.device
            )
            kv_mask = input_mask.bool()
            attention_mask = make_cross_attention_mask(query_mask, kv_mask)

        # Apply Self Attention layers across all blocks
        kv_cache = None
        for i in range(self.num_blocks):
            # Apply Cross Attention
            if i == 0:
                z, kv_cache = self.cross_attend_0(
                    inputs_q=z,
                    inputs_kv=inputs,
                    attention_mask=attention_mask,
                    is_training=self.training,
                    kv_cache=None,
                )
            elif i == 1:
                z, kv_cache = self.cross_attend_1(
                    inputs_q=z,
                    inputs_kv=inputs,  # Still use embeddings for the second layer
                    attention_mask=attention_mask,
                    is_training=self.training,
                    kv_cache=None,  # kv_cache is still None here according to the pseudocode
                )
            else:
                z, kv_cache = self.cross_attend_1(
                    inputs_q=z,
                    inputs_kv=None,  # Important: context is None
                    attention_mask=attention_mask,
                    is_training=self.training,
                    kv_cache=kv_cache,
                )
            for self_attend in self.self_attends:
                z = self_attend(
                    inputs=z, attention_mask=None, is_training=self.training
                )

        if hasattr(self, "output_projection"):
            z = self.output_projection(z)

        return z
