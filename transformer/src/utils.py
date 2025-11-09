"""
Utility functions for Transformer
"""
import torch
import torch.nn as nn


def create_padding_mask(seq, pad_idx=0):
    """
    Create padding mask
    
    Args:
        seq: Input sequence [batch_size, seq_len]
        pad_idx: Padding token index
    
    Returns:
        mask: Padding mask [batch_size, 1, seq_len]
    """
    mask = (seq != pad_idx).unsqueeze(1)
    return mask.float()


def create_look_ahead_mask(size):
    """
    Create look-ahead mask for decoder (prevents attending to future positions)
    
    Args:
        size: Sequence length
    
    Returns:
        mask: Look-ahead mask [size, size]
    """
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask == 0


def create_decoder_mask(tgt, pad_idx=0):
    """
    Create combined mask for decoder (padding + look-ahead)
    
    Args:
        tgt: Target sequence [batch_size, tgt_seq_len]
        pad_idx: Padding token index
    
    Returns:
        mask: Combined mask [batch_size, tgt_seq_len, tgt_seq_len]
    """
    batch_size, seq_len = tgt.size()
    
    device = tgt.device
    
    # Padding mask
    padding_mask = create_padding_mask(tgt, pad_idx).to(device).bool()  # [batch_size, 1, seq_len]
    
    # Look-ahead mask
    look_ahead_mask = create_look_ahead_mask(seq_len).to(device)  # [seq_len, seq_len]
    look_ahead_mask = look_ahead_mask.unsqueeze(0).expand(batch_size, -1, -1)
    
    # Combine masks
    mask = padding_mask & look_ahead_mask
    
    return mask.float()


def create_lm_mask(seq, pad_idx=0):
    """
    Create causal mask for language modeling (padding + look-ahead)
    
    Args:
        seq: Input sequence [batch_size, seq_len]
        pad_idx: Padding token index
    
    Returns:
        mask: Causal mask [batch_size, seq_len, seq_len]
    """
    batch_size, seq_len = seq.size()
    device = seq.device
    
    # Padding mask
    padding_mask = (seq != pad_idx).unsqueeze(1).expand(-1, seq_len, -1)
    
    # Look-ahead mask
    look_ahead_mask = create_look_ahead_mask(seq_len).to(device)
    look_ahead_mask = look_ahead_mask.unsqueeze(0).expand(batch_size, -1, -1)
    
    mask = padding_mask & look_ahead_mask
    return mask.float()


def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(model):
    """Initialize model weights using Xavier uniform initialization"""
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)


