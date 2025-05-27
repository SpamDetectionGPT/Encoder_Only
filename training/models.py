import torch
from torch import nn
import torch.nn.functional as F
from math import sqrt


def scaled_dot_product_attention(query, key, value):
    """
    Compute scaled dot product attention: Attention(Q,K,V) = softmax(QK^T/√d_k)V

    Args:
        query (torch.Tensor): Query tensor of shape (batch_size, seq_len_q, 64).
        key (torch.Tensor): Key tensor of shape (batch_size, seq_len_k, 64).
        value (torch.Tensor): Value tensor of shape (batch_size, seq_len_v, 64).

    Returns:
        torch.Tensor: Attention output of shape (batch_size, seq_len_q, 64).
        numpy.ndarray: Attention matrix for visualization.
    """
    # Compute attention scores and scale by sqrt(d_k)
    dim_k = key.size(-1)  # 64 for BERT-base
    scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(dim_k)

    # Apply softmax to get attention weights
    weights = F.softmax(scores, dim=-1)

    # Save attention matrix for visualization
    att_mat = weights.detach().cpu().numpy()

    # Apply attention weights to values
    return torch.bmm(weights, value), att_mat


class AttentionHead(nn.Module):
    """
    Single attention head with Q, K, V projections and scaled dot-product attention.

    Transforms input (768-dim) to Q, K, V (64-dim each) and computes attention.

    Args:
        embed_dim (int): Input embedding dimension (768 for BERT-base).
        head_dim (int): Head dimension (64 for BERT-base).
    """

    def __init__(self, embed_dim, head_dim):
        super().__init__()
        # Linear projections: 768 → 64 for BERT-base
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)
        self.att_mat = None

    def forward(self, hidden_state):
        """
        Forward pass through attention head.

        Args:
            hidden_state (torch.Tensor): Input of shape (batch_size, seq_len, 768).

        Returns:
            torch.Tensor: Attention output of shape (batch_size, seq_len, 64).
        """
        # Apply Q, K, V projections
        query = self.q(hidden_state)
        key = self.k(hidden_state)
        value = self.v(hidden_state)

        # Compute attention
        attn_outputs, attn_mat = scaled_dot_product_attention(query, key, value)
        self.att_mat = attn_mat

        return attn_outputs


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention with 12 parallel attention heads (BERT-base).

    Each head processes 768→64 dims, outputs are concatenated and projected back to 768.

    Args:
        config: Configuration with hidden_size=768, num_attention_heads=12.
    """

    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size  # 768
        num_heads = config.num_attention_heads  # 12
        head_dim = embed_dim // num_heads  # 64

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        # Create 12 attention heads
        self.heads = nn.ModuleList(
            [AttentionHead(embed_dim, head_dim) for _ in range(num_heads)]
        )
        self.att_mats = [i.att_mat for i in self.heads]

        # Output projection: 768 → 768
        self.output_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, hidden_state):
        """
        Forward pass through multi-head attention.

        Args:
            hidden_state (torch.Tensor): Input of shape (batch_size, seq_len, 768).

        Returns:
            torch.Tensor: Output of shape (batch_size, seq_len, 768).
        """
        # Process through all heads in parallel
        head_outputs = [h(hidden_state) for h in self.heads]

        # Concatenate: 12×64 = 768
        concatenated_output = torch.cat(head_outputs, dim=-1)

        # Final projection
        return self.output_linear(concatenated_output)


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network: 768 → 3072 → 768 with GELU activation.

    Args:
        config: Configuration with hidden_size=768, intermediate_size=3072.
    """

    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(
            config.hidden_size, config.intermediate_size
        )  # 768 → 3072
        self.linear2 = nn.Linear(
            config.intermediate_size, config.hidden_size
        )  # 3072 → 768
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        """
        Forward pass through feed-forward network.

        Args:
            x (torch.Tensor): Input of shape (batch_size, seq_len, 768).

        Returns:
            torch.Tensor: Output of shape (batch_size, seq_len, 768).
        """
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """
    Single Transformer encoder layer with Pre-LayerNorm architecture.

    Contains: LayerNorm → MultiHeadAttention → Residual → LayerNorm → FeedForward → Residual

    Args:
        config: Configuration with hidden_size=768.
    """

    def __init__(self, config):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm_2 = nn.LayerNorm(config.hidden_size)
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)

    def forward(self, x):
        """
        Forward pass through encoder layer.

        Args:
            x (torch.Tensor): Input of shape (batch_size, seq_len, 768).

        Returns:
            torch.Tensor: Output of shape (batch_size, seq_len, 768).
        """
        # Attention sub-layer with residual connection
        hidden_state = self.layer_norm_1(x)
        attention_output = self.attention(hidden_state)
        x = x + attention_output

        # Feed-forward sub-layer with residual connection
        normalized_x = self.layer_norm_2(x)
        ffn_output = self.feed_forward(normalized_x)
        x = x + ffn_output

        return x


class Embeddings(nn.Module):
    """
    Token and position embeddings with layer normalization and dropout.

    Converts token IDs to dense vectors and adds positional information.

    Args:
        config: Configuration with vocab_size=30522, hidden_size=768, max_position_embeddings=512.
    """

    def __init__(self, config):
        super().__init__()
        if not hasattr(config, "vocab_size"):
            raise AttributeError("Config object must have 'vocab_size' attribute.")

        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )

        layer_norm_eps = getattr(config, "layer_norm_eps", 1e-12)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=layer_norm_eps)

        dropout_prob = getattr(config, "hidden_dropout_prob", 0.0)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input_ids):
        """
        Forward pass through embeddings.

        Args:
            input_ids (torch.Tensor): Token IDs of shape (batch_size, seq_length).

        Returns:
            torch.Tensor: Embeddings of shape (batch_size, seq_length, 768).
        """
        seq_length = input_ids.size(1)

        # Create position IDs [0, 1, 2, ..., seq_length-1]
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device
        ).unsqueeze(0)

        # Get token and position embeddings
        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        # Combine and normalize
        embeddings = token_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class TransformerEncoder(nn.Module):
    """
    Complete Transformer encoder with embeddings and 12 encoder layers (BERT-base).

    Args:
        config: Configuration with num_hidden_layers=12.
    """

    def __init__(self, config):
        super().__init__()
        self.embeddings = Embeddings(config)

        if not hasattr(config, "num_hidden_layers"):
            raise AttributeError(
                "Config object must have 'num_hidden_layers' attribute."
            )

        # Stack of 12 encoder layers
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, input_ids):
        """
        Forward pass through complete encoder.

        Args:
            input_ids (torch.Tensor): Token IDs of shape (batch_size, seq_length).

        Returns:
            torch.Tensor: Contextualized representations of shape (batch_size, seq_length, 768).
        """
        # Convert token IDs to embeddings
        hidden_states = self.embeddings(input_ids)

        # Process through all encoder layers
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        return hidden_states


class TransformerForSequenceClassification(nn.Module):
    """
    Complete Transformer model for sequence classification.

    Uses [CLS] token representation for classification after encoder processing.

    Args:
        config: Configuration with hidden_size=768, num_labels=2.
    """

    def __init__(self, config):
        super().__init__()
        self.encoder = TransformerEncoder(config)

        dropout_prob = getattr(config, "hidden_dropout_prob", 0.0)
        self.dropout = nn.Dropout(dropout_prob)

        if not hasattr(config, "num_labels"):
            raise AttributeError("Config object must have 'num_labels' attribute.")

        # Classification head: 768 → num_labels
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids):
        """
        Forward pass through complete model.

        Args:
            input_ids (torch.Tensor): Token IDs of shape (batch_size, seq_length).

        Returns:
            torch.Tensor: Classification logits of shape (batch_size, num_labels).
        """
        # Encode input sequence
        encoder_output = self.encoder(input_ids)

        # Extract [CLS] token (first token)
        pooled_output = encoder_output[:, 0, :]

        # Apply dropout and classify
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits
