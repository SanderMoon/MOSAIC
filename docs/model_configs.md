# Model configuration

## Overview

This documentation provides descriptions of the configuration fields required to set up the CustomCoCa model. The configurations cover the vision encoder, text encoder, and multimodal decoder.

## General Configuration

### embed_dim

- **Type**: `int`
- **Description**: The dimensionality of the embeddings used in the model.
  
  ```json
  "embed_dim": 1024
  ```

## Vision Configuration

### vision_cfg

Configuration settings for the Vision Encoder.

```json
"vision_cfg": {
    "input_dim": 192,
    "method": "vit",
    "vit_config": {
        "vit_patch_size": 16,
        "vit_mlp_ratio": 4,
        "vit_layers": 2,
        "vit_heads": 4
    },
    "pooling_strategy": "attentional",
    "attentional_pool_vector": true,
    "attn_pooler_vector_heads": 8,
    "attentional_pool_matrix": true,
    "attn_pooler_matrix_length": 128,
    "attn_pooler_matrix_heads": 8
}
```

#### Fields

- **input_dim**:
  - **Type**: `int`
  - **Description**: The input dimension of the visual data (preprocessed feature vector dimensions).

- **method**:
  - **Type**: `str`
  - **Description**: The method used for the vision encoder. Default is 'vit' (Vision Transformer).

- **vit_config**:
  - **Type**: `object`
  - **Description**: Configuration for the Vision Transformer (ViT).
  - **Fields**:
    - **vit_mlp_ratio**:
      - **Type**: `int`
      - **Description**: The ratio of the MLP (Multi-Layer Perceptron) units.

    - **vit_layers**:
      - **Type**: `int`
      - **Description**: The number of layers in the ViT.

    - **vit_heads**:
      - **Type**: `int`
      - **Description**: The number of attention heads in the ViT.

- **pooling_strategy**:
  - **Type**: `str`
  - **Description**: Strategy for pooling visual features. Default is 'attentional'.

- **attentional_pool_vector**:
  - **Type**: `bool`
  - **Description**: Whether to use attentional pooling for creating a single image embedding.

- **attn_pooler_vector_heads**:
  - **Type**: `int`
  - **Description**: Number of attention heads for attentional vector pooling.

- **attentional_pool_matrix**:
  - **Type**: `bool`
  - **Description**: Whether to use attentional pooling to reduce the set of output embeddings for decoding.

- **attn_pooler_matrix_length**:
  - **Type**: `int`
  - **Description**: The number of rows in the learnable matrix for attentional pooling. (determines the number of latent vectors for decoding).

- **attn_pooler_matrix_heads**:
  - **Type**: `int`
  - **Description**: Number of attention heads for matrix pooling.

## Text Configuration

### text_cfg

Configuration settings for the Text Encoder.

```json
"text_cfg": {
    "context_length": 76,
    "load_pretrained": true,
    "hf_model_name": "microsoft/biogpt",
    "hf_tokenizer_name": "microsoft/biogpt",
    "load_pretrained_model_start_layer": 0,
    "load_pretrained_model_end_layer": 12,
    "vocab_size": 42384,
    "heads": 16,
    "layers": 12,
    "freeze_cls_embed": true,
    "freeze_base": true,
    "lora": true,
    "lora_config": {
        "r": 4,
        "lora_alpha": 4,
        "target_modules": ["q_proj", "k_proj", "v_proj"],
        "lora_dropout": 0,
        "bias": "none"
    }
}
```

#### Fields

- **load_pretrained**:
  - **Type**: `bool`
  - **Description**: Whether to load a pretrained model.

- **hf_model_name**:
  - **Type**: `str`
  - **Description**: The name of the Hugging Face model to use.

- **hf_tokenizer_name**:
  - **Type**: `str`
  - **Description**: The name of the Hugging Face tokenizer to use.

- **load_pretrained_model_start_layer**:
  - **Type**: `int`
  - **Description**: The starting layer index for loading a pretrained model.

- **load_pretrained_model_end_layer**:
  - **Type**: `int`
  - **Description**: The ending layer index for loading a pretrained model.

- **vocab_size**:
  - **Type**: `int`
  - **Description**: Size of the vocabulary.

- **heads**:
  - **Type**: `int`
  - **Description**: Number of attention heads in the text model.

- **layers**:
  - **Type**: `int`
  - **Description**: Number of Transformer layers in the text model.

- **freeze_cls_embed**:
  - **Type**: `bool`
  - **Description**: Whether to embed the class token.

- **freeze_base**:
  - **Type**: `bool`
  - **Description**: Whether to freeze the base encoder layers.

- **lora**:
  - **Type**: `bool`
  - **Description**: Whether to use LoRA (Low-Rank Adaptation).

- **lora_config**:
  - **Type**: `object`
  - **Description**: Configuration for LoRA (Low-Rank Adaptation).
  - **Fields**:
    - **r**:
      - **Type**: `int`
      - **Description**: Rank for the LoRA adaptation.

    - **lora_alpha**:
      - **Type**: `int`
      - **Description**: LoRA scaling factor.

    - **target_modules**:
      - **Type**: `list[str]`
      - **Description**: List of target modules for LoRA.

    - **lora_dropout**:
      - **Type**: `float`
      - **Description**: Dropout rate for LoRA.

    - **bias**:
      - **Type**: `str`
      - **Description**: Type of bias to use.

## Multimodal Configuration

### multimodal_cfg

Configuration settings for the Multimodal Decoder.

```json
"multimodal_cfg": {
    "context_length": 76,
    "hf_model_name": "microsoft/biogpt",
    "load_pretrained_model_start_layer": 12,
    "load_pretrained_model_end_layer": 24,
    "vocab_size": 42384,
    "heads": 16,
    "layers": 12,
    "cross_attention_type": "default",
    "load_pretrained": true,
    "freeze_base": true,
    "freeze_cross_attn": false
}
```

#### Fields

- **context_length**:
  - **Type**: `int`
  - **Description**: Length of the context in tokens.

- **hf_model_name**:
  - **Type**: `str`
  - **Description**: The name of the Hugging Face model to use.

- **load_pretrained_model_start_layer**:
  - **Type**: `int`
  - **Description**: The starting layer index for loading a pretrained model.

- **load_pretrained_model_end_layer**:
  - **Type**: `int`
  - **Description**: The ending layer index for loading a pretrained model.

- **vocab_size**:
  - **Type**: `int`
  - **Description**: Size of the vocabulary.

- **heads**:
  - **Type**: `int`
  - **Description**: Number of attention heads in the multimodal model.

- **layers**:
  - **Type**: `int`
  - **Description**: Number of Transformer layers in the multimodal model.

- **cross_attention_type**:
  - **Type**: `str`
  - **Description**: The type of cross-attention mechanism.

- **load_pretrained**:
  - **Type**: `bool`
  - **Description**: Whether to load a pretrained model.

- **freeze_base**:
  - **Type**: `bool`
  - **Description**: Whether to freeze the base encoder layers.

- **freeze_cross_attn**:
  - **Type**: `bool`
  - **Description**: Whether to freeze the cross-attention layers.

## Example Configuration

Here is the full example configuration file for the CustomCoCa model:

```json
{
    "embed_dim": 1024,
    "vision_cfg": {
        "input_dim": 192,
        "method": "vit",
        "vit_config": {
            "vit_patch_size": 16,
            "vit_mlp_ratio": 4,
            "vit_layers": 2,
            "vit_heads": 4
        },
        "pooling_strategy": "attentional",
        "attentional_pool_vector": true,
        "attn_pooler_vector_heads": 8,
        "attentional_pool_matrix": true,
        "attn_pooler_matrix_length": 128,
        "attn_pooler_matrix_heads": 8
    },
    "text_cfg": {
        "context_length": 76,
        "load_pretrained": true,
        "hf_model_name": "microsoft/biogpt",
        "hf_tokenizer_name": "microsoft/biogpt",
        "load_pretrained_model_start_layer": 0,
        "load_pretrained_model_end_layer": 12,
        "vocab_size": 42384,
        "heads": 16,
        "layers": 12,
        "freeze_cls_embed": true,
        "freeze_base": true,
        "lora": true,
        "lora_config": {
            "r": 4,
            "lora_alpha": 4,
            "target_modules": ["q_proj", "k_proj", "v_proj"],
            "lora_dropout": 0,
            "bias": "none"
        }
    },
    "multimodal_cfg": {
        "context_length": 76,
        "hf_model_name": "microsoft/biogpt",
        "load_pretrained_model_start_layer": 12,
        "load_pretrained_model_end_layer": 24,
        "vocab_size": 42384,
        "heads": 16,
        "layers": 12,
        "cross_attention_type": "default",
        "load_pretrained": true,
        "freeze_base": true,
        "freeze_cross_attn": false
    }
}
```

