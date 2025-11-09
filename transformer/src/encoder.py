"""
Transformer Encoder Implementation
"""
import math
import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .feed_forward import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    """
    Single Transformer Encoder Layer
    
    Consists of:
    1. Multi-head self-attention
    2. Residual connection + LayerNorm
    3. Position-wise feed-forward network
    4. Residual connection + LayerNorm
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Padding mask [batch_size, 1, seq_len] or [batch_size, seq_len, seq_len]
        
        Returns:
            output: Encoder layer output [batch_size, seq_len, d_model]
            attn_weights: Attention weights
        """
        # Multi-head self-attention with residual connection
        attn_output, attn_weights = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x, attn_weights


class TransformerEncoder(nn.Module):
    """
    Complete Transformer Encoder
    
    Args:
        vocab_size: vocabulary size
        d_model: model dimension
        num_layers: number of encoder layers
        num_heads: number of attention heads
        d_ff: feed-forward dimension
        max_len: maximum sequence length
        dropout: dropout probability
        use_learned_pos: whether to use learned positional encoding
    """
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff,
                 max_len=5000, dropout=0.1, use_learned_pos=False):
        super().__init__()
        self.d_model = d_model
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        if use_learned_pos:
            from .positional_encoding import LearnedPositionalEncoding
            self.pos_encoding = LearnedPositionalEncoding(d_model, max_len, dropout)
        else:
            from .positional_encoding import PositionalEncoding
            self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Encoder layers
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input token indices [batch_size, seq_len]
            mask: Padding mask [batch_size, 1, seq_len] or [batch_size, seq_len, seq_len]
        
        Returns:
            output: Encoder output [batch_size, seq_len, d_model]
            all_attn_weights: List of attention weights from each layer
        """
        # Embedding and positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        # Pass through encoder layers
        all_attn_weights = []
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            all_attn_weights.append(attn_weights)
        
        return x, all_attn_weights

