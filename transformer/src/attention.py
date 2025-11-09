"""
Scaled Dot-Product Attention and Multi-Head Attention Implementation
"""
import torch
import torch.nn as nn
import math


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention
    
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    
    Args:
        d_k: dimension of key/query
        dropout: dropout probability
    """
    def __init__(self, d_k, dropout=0.1):
        super().__init__()
        self.d_k = d_k
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q: Query tensor [batch_size, num_heads, seq_len, d_k] or [batch_size, seq_len, d_k]
            K: Key tensor [batch_size, num_heads, seq_len, d_k] or [batch_size, seq_len, d_k]
            V: Value tensor [batch_size, num_heads, seq_len, d_v] or [batch_size, seq_len, d_v]
            mask: Mask tensor [batch_size, seq_len, seq_len] or [batch_size, 1, seq_len]
        
        Returns:
            output: Attention output [batch_size, num_heads, seq_len, d_v] or [batch_size, seq_len, d_v]
            attn_weights: Attention weights [batch_size, num_heads, seq_len, seq_len] or [batch_size, seq_len, seq_len]
        """
        # Compute attention scores: QK^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            # Handle different mask shapes
            if mask.dim() == 3:
                # mask: [batch_size, seq_len, seq_len] or [batch_size, 1, seq_len]
                if mask.size(1) == 1:
                    # [batch_size, 1, seq_len] -> expand to [batch_size, seq_len, seq_len]
                    mask = mask.expand(-1, scores.size(-2), -1)
                # If scores has num_heads dimension, expand mask
                if scores.dim() == 4:  # [batch_size, num_heads, seq_len, seq_len]
                    mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]
            elif mask.dim() == 2:
                # [seq_len, seq_len] -> expand
                if scores.dim() == 4:
                    mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
                else:
                    mask = mask.unsqueeze(0)  # [1, seq_len, seq_len]
            
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, V)
        
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention
    
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
    where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    
    Args:
        d_model: model dimension
        num_heads: number of attention heads
        dropout: dropout probability
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(self.d_k, dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q: Query tensor [batch_size, seq_len, d_model]
            K: Key tensor [batch_size, seq_len, d_model]
            V: Value tensor [batch_size, seq_len, d_model]
            mask: Mask tensor
        
        Returns:
            output: Multi-head attention output [batch_size, seq_len, d_model]
            attn_weights: Attention weights [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size = Q.size(0)
        q_len = Q.size(1)
        k_len = K.size(1)
        v_len = V.size(1)
        
        # Linear projections and split into heads
        Q = self.W_q(Q).view(batch_size, q_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, k_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, v_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention to each head
        # Q, K, V: [batch_size, num_heads, seq_len, d_k]
        output, attn_weights = self.attention(Q, K, V, mask)
        
        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(
            batch_size, q_len, self.d_model
        )
        
        # Final linear projection
        output = self.W_o(output)
        
        return output, attn_weights

