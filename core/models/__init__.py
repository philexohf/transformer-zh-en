from .transformer import Transformer, MultiHeadAttention, PositionalEncoding, AddNorm

def build_model(vocab_size, config):
    """构建Transformer模型 - 兼容接口"""
    return Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=config.d_model,
        num_heads=config.nhead,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        d_ffn=config.d_ff,
        dropout=config.dropout,
        max_len=config.max_len,
        pad_idx=0
    )