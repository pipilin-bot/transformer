"""
Complete Transformer Model (Encoder-Decoder Architecture)
"""
import torch
import torch.nn as nn
from .encoder import TransformerEncoder
from .decoder import TransformerDecoder


class Transformer(nn.Module):
    """
    Complete Transformer Model (Encoder-Decoder)
    
    Args:
        src_vocab_size: source vocabulary size
        tgt_vocab_size: target vocabulary size
        d_model: model dimension
        num_layers: number of encoder/decoder layers
        num_heads: number of attention heads
        d_ff: feed-forward dimension
        max_len: maximum sequence length
        dropout: dropout probability
        use_learned_pos: whether to use learned positional encoding
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_layers,
                 num_heads, d_ff, max_len=5000, dropout=0.1, use_learned_pos=False):
        super().__init__()
        
        self.encoder = TransformerEncoder(
            src_vocab_size, d_model, num_layers, num_heads, d_ff,
            max_len, dropout, use_learned_pos
        )
        
        self.decoder = TransformerDecoder(
            tgt_vocab_size, d_model, num_layers, num_heads, d_ff,
            max_len, dropout, use_learned_pos
        )
        
        # Output projection
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Args:
            src: Source token indices [batch_size, src_seq_len]
            tgt: Target token indices [batch_size, tgt_seq_len]
            src_mask: Source padding mask [batch_size, 1, src_seq_len]
            tgt_mask: Target mask (padding + future) [batch_size, tgt_seq_len, tgt_seq_len]
        
        Returns:
            output: Logits [batch_size, tgt_seq_len, tgt_vocab_size]
            encoder_attn_weights: Encoder attention weights
            decoder_self_attn_weights: Decoder self-attention weights
            decoder_cross_attn_weights: Decoder cross-attention weights
        """
        # Encode
        encoder_output, encoder_attn_weights = self.encoder(src, src_mask)
        
        # Decode
        decoder_output, decoder_self_attn_weights, decoder_cross_attn_weights = \
            self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        
        # Project to vocabulary
        output = self.output_proj(decoder_output)
        
        return output, encoder_attn_weights, decoder_self_attn_weights, decoder_cross_attn_weights


class TransformerLM(nn.Module):
    """
    Transformer Language Model (Encoder-only for language modeling)
    
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
        
        self.encoder = TransformerEncoder(
            vocab_size, d_model, num_layers, num_heads, d_ff,
            max_len, dropout, use_learned_pos
        )
        
        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input token indices [batch_size, seq_len]
            mask: Padding mask [batch_size, 1, seq_len]
        
        Returns:
            output: Logits [batch_size, seq_len, vocab_size]
            attn_weights: Attention weights from all layers
        """
        encoder_output, attn_weights = self.encoder(x, mask)
        output = self.output_proj(encoder_output)
        return output, attn_weights



