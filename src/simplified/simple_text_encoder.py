import torch
from torch import nn
from transformers import BertModel, BertTokenizer

from mosaic.attentional_pooling import AttentionalPooling

# Assuming AttentionalPooling is defined as provided
# from your_module import AttentionalPooling


class TextEncoder(nn.Module):
    """
    Text Encoder that encodes captions using a frozen BERT model followed by attentional pooling.

    Args:
        bert_model_name (str): Name of the pre-trained BERT model (e.g., 'bert-base-uncased').
        pooler_dim (int, optional): Dimensionality for the attentional pooling query. Defaults to BERT's hidden size.
        num_heads (int, optional): Number of attention heads in AttentionalPooling. Defaults to 8.
    """

    def __init__(
        self, bert_model_name="bert-base-uncased", num_heads=8, output_dim=1024
    ):
        super(TextEncoder, self).__init__()

        # Load pre-trained BERT model
        self.bert = BertModel.from_pretrained(bert_model_name)

        # Freeze BERT parameters
        for param in self.bert.parameters():
            param.requires_grad = False

        # BERT's hidden size
        hidden_size = self.bert.config.hidden_size

        # upsample the input features to the common dim
        self.phi = nn.Sequential(nn.Linear(hidden_size, output_dim), nn.GELU())

        # Initialize a single learnable query vector
        # Shape: [1, 1, D]
        self.query = nn.Parameter(torch.randn(1, 1, output_dim))

        # Attentional Pooling module
        self.attentional_pooling = AttentionalPooling(
            dim=output_dim, num_heads=num_heads
        )

    def forward(self, input_ids, attention_mask):
        """
        Forward pass for the TextEncoder.

        Args:
            input_ids (torch.Tensor): Tokenized input IDs of shape (B, L).
            attention_mask (torch.Tensor): Attention mask of shape (B, L), where 1 indicates valid tokens.

        Returns:
            torch.Tensor: Pooled text embeddings of shape (B, D).
        """
        # Obtain token-level embeddings from BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state  # Shape: [B, L, D]

        B, L, D = token_embeddings.size()

        # upsample the input features to the common dim
        token_embeddings = self.phi(token_embeddings)

        # Expand the learnable query to match the batch size
        # Shape: [B, 1, D]
        expanded_query = self.query.expand(B, -1, -1)

        # Apply attentional pooling
        # Output shape: [B, 1, D]
        pooled_output = self.attentional_pooling(
            features=token_embeddings,
            query=expanded_query,
            attention_mask=attention_mask,
        )

        # Squeeze the sequence dimension to get [B, D]
        pooled_output = pooled_output.squeeze(1)

        return pooled_output


# Example Usage
if __name__ == "__main__":
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Define dummy captions
    captions = [
        "A cat sitting on a mat.",
        "A dog playing with a ball.",
        "A bird flying in the sky.",
        "A horse running in the field.",
    ]

    # Tokenize captions
    encoding = tokenizer(
        captions, return_tensors="pt", padding=True, truncation=True, max_length=20
    )

    input_ids = encoding["input_ids"]  # Shape: [4, L]
    attention_mask = encoding["attention_mask"]  # Shape: [4, L]

    # Initialize TextEncoder
    text_encoder = TextEncoder(bert_model_name="bert-base-uncased", num_heads=8)

    # Forward pass
    text_embeddings = text_encoder(input_ids, attention_mask)
    print(f"Text Embeddings Shape: {text_embeddings.shape}")  # Expected: [4, 768]
