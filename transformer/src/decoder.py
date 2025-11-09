"""
Transformer Decoder Implementation
"""
import math
import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .feed_forward import PositionwiseFeedForward


class DecoderLayer(nn.Module):
    """
    Single Transformer Decoder Layer
    
    Consists of:
    1. Masked multi-head self-attention
    2. Residual connection + LayerNorm
    3. Multi-head cross-attention (encoder-decoder attention)
    4. Residual connection + LayerNorm
    5. Position-wise feed-forward network
    6. Residual connection + LayerNorm
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Args:
            x: Decoder input tensor [batch_size, tgt_seq_len, d_model]
            encoder_output: Encoder output [batch_size, src_seq_len, d_model]
            src_mask: Source mask (padding mask) [batch_size, 1, src_seq_len]
            tgt_mask: Target mask (padding + future mask) [batch_size, tgt_seq_len, tgt_seq_len]
        
        Returns:
            output: Decoder layer output [batch_size, tgt_seq_len, d_model]
            self_attn_weights: Self-attention weights
            cross_attn_weights: Cross-attention weights
        """
        # Masked self-attention
        self_attn_output, self_attn_weights = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(self_attn_output))
        
        # Cross-attention (encoder-decoder attention)
        cross_attn_output, cross_attn_weights = self.cross_attn(
            x, encoder_output, encoder_output, src_mask
        )
        x = self.norm2(x + self.dropout2(cross_attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))
        
        return x, self_attn_weights, cross_attn_weights


class TransformerDecoder(nn.Module):
    """
    Complete Transformer Decoder
    
    Args:
        vocab_size: vocabulary size
        d_model: model dimension
        num_layers: number of decoder layers
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
        
        # Decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Args:
            x: Target token indices [batch_size, tgt_seq_len]
            encoder_output: Encoder output [batch_size, src_seq_len, d_model]
            src_mask: Source padding mask [batch_size, 1, src_seq_len]
            tgt_mask: Target mask (padding + future) [batch_size, tgt_seq_len, tgt_seq_len]
        
        Returns:
            output: Decoder output [batch_size, tgt_seq_len, d_model]
            all_self_attn_weights: List of self-attention weights
            all_cross_attn_weights: List of cross-attention weights
        """
        # Embedding and positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        # Pass through decoder layers
        all_self_attn_weights = []
        all_cross_attn_weights = []
        for layer in self.layers:
            x, self_attn_weights, cross_attn_weights = layer(
                x, encoder_output, src_mask, tgt_mask
            )
            all_self_attn_weights.append(self_attn_weights)
            all_cross_attn_weights.append(cross_attn_weights)
        
        return x, all_self_attn_weights, all_cross_attn_weights



