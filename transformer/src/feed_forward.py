"""
Position-wise Feed-Forward Network Implementation
"""
import torch
import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network
    
    FFN(x) = max(0, xW1 + b1)W2 + b2
    
    Args:
        d_model: model dimension
        d_ff: feed-forward dimension (typically 4 * d_model)
        dropout: dropout probability
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # or nn.ReLU()
        
    def forward(self, x):
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
        
        Returns:
            output: FFN output [batch_size, seq_len, d_model]
        """
        return self.linear2(self.dropout(self.activation(self.linear1(x))))



