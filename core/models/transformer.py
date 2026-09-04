"""
Transformer 模型实现（Vaswani et al., 2017）

Base: d_model=512, heads=8, layers=6, FFN=2048, ~93M params
"""

import math
import torch
from torch import nn
import torch.nn.functional as F


# --- 基础组件

class PositionalEncoding(nn.Module):
    """
    正弦位置编码 — Vaswani et al., Sec 3.5

    PE(pos, 2i)   = sin(pos / 10000^{2i/d_model})
    PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})

    正弦函数的线性性质使模型可学习相对位置关系。
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 预计算位置编码，注册为 buffer（不参与梯度更新）
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        if x.size(1) > self.pe.size(1):
            pe = torch.zeros(x.size(1), self.pe.size(-1), device=x.device)
            position = torch.arange(0, x.size(1), dtype=torch.float, device=x.device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.pe.size(-1), 2, device=x.device).float() * (-math.log(10000.0) / self.pe.size(-1)))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            x = x + pe[:, :x.size(1), :]
        else:
            x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


def scaled_dot_product_attention(query, key, value, mask=None):
    """
    缩放点积注意力 — Vaswani et al., Eq. 1
    Attention(Q,K,V) = softmax(QK^T / √d_k)V
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))

    attention_weights = F.softmax(scores, dim=-1)

    # 全 mask 行 softmax 会返回 NaN，替换为 0
    attention_weights = torch.where(torch.isnan(attention_weights),
                                    torch.zeros_like(attention_weights),
                                    attention_weights)
    return torch.matmul(attention_weights, value), attention_weights


class MultiHeadAttention(nn.Module):
    """
    多头注意力 — Vaswani et al., Sec 3.2.2

    将 d_model 拆为 h 个子空间，各头独立做注意力后拼接。
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_v).transpose(1, 2)

        x, attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        x = self.dropout(x)
        return self.W_o(x), attn_weights


# --- 前馈网络 & 残差连接

class PositionWiseFFN(nn.Module):
    """
    逐位置前馈网络 — Vaswani et al., Eq. 2
    FFN(x) = ReLU(xW1 + b1)W2 + b2
    d_model -> d_ffn -> d_model（在各位置独立应用）
    """
    def __init__(self, d_model, d_ffn, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
        )

    def forward(self, X):
        return self.net(X)


class AddNorm(nn.Module):
    """
    残差连接 + 层归一化 — Vaswani et al., Sec 5.4
    output = LayerNorm(x + Dropout(Sublayer(x)))
    Post-LN 架构（论文原版）。
    """
    def __init__(self, normalized_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, residual, sublayer_output):
        return self.ln(residual + self.dropout(sublayer_output))


# --- 编码器 & 解码器层

class EncoderLayer(nn.Module):
    """
    编码器层：Self-Attention + FFN，各带残差连接。
    """
    def __init__(self, d_model, num_heads, d_ffn, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionWiseFFN(d_model, d_ffn, dropout)
        self.add_norm1 = AddNorm(d_model, dropout)
        self.add_norm2 = AddNorm(d_model, dropout)

    def forward(self, X, mask=None):
        attn_output, _ = self.self_attn(X, X, X, mask)
        X = self.add_norm1(X, attn_output)
        ffn_output = self.ffn(X)
        X = self.add_norm2(X, ffn_output)
        return X


class DecoderLayer(nn.Module):
    """
    解码器层：Masked Self-Attn + Cross-Attn + FFN，各带残差连接。
    """
    def __init__(self, d_model, num_heads, d_ffn, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionWiseFFN(d_model, d_ffn, dropout)
        self.add_norm1 = AddNorm(d_model, dropout)
        self.add_norm2 = AddNorm(d_model, dropout)
        self.add_norm3 = AddNorm(d_model, dropout)

    def forward(self, X, enc_output, tgt_mask=None, src_mask=None):
        attn_output, _ = self.self_attn(X, X, X, tgt_mask)
        X = self.add_norm1(X, attn_output)
        cross_attn_output, _ = self.cross_attn(X, enc_output, enc_output, src_mask)
        X = self.add_norm2(X, cross_attn_output)
        ffn_output = self.ffn(X)
        X = self.add_norm3(X, ffn_output)
        return X


# --- Transformer 模型

class Transformer(nn.Module):
    """
    完整 Transformer 模型 — Vaswani et al., Sec 3.1

    三个入口：forward() 训练, encode() 编码, decode() 解码（支持逐步生成）
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8,
                 num_encoder_layers=6, num_decoder_layers=6, d_ffn=2048, dropout=0.1,
                 max_len=5000, pad_idx=0):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx

        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        self.encoder = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ffn, dropout)
                          for _ in range(num_encoder_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ffn, dropout)
                          for _ in range(num_decoder_layers)])
        self.linear = nn.Linear(d_model, tgt_vocab_size)
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform 初始化 — Vaswani et al., Sec 3.2.2"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # -- 掩码生成

    def generate_square_subsequent_mask(self, sz):
        """
        因果掩码 — 下三角矩阵，True 表示可见。
        """
        mask = torch.tril(torch.ones(sz, sz)).bool()
        return mask

    def make_src_mask(self, src):
        """padding 掩码，返回 [batch, 1, 1, src_len] 支持广播。"""
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_tgt_mask(self, tgt):
        """目标掩码 = 因果掩码 & padding 掩码。"""
        tgt_len = tgt.size(1)
        subsequent_mask = self.generate_square_subsequent_mask(tgt_len).to(tgt.device)
        padding_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)
        return padding_mask & subsequent_mask

    # -- 前向传播

    def encode(self, src):
        src_padding_mask = self.make_src_mask(src)
        src_mask = src_padding_mask & src_padding_mask.transpose(-2, -1)

        src_embed = self.src_embed(src) * math.sqrt(self.d_model)
        src_embed = self.pos_encoder(src_embed)

        enc_output = src_embed
        for layer in self.encoder:
            enc_output = layer(enc_output, src_mask)
        return enc_output, src_padding_mask

    def decode(self, tgt, encoder_output, src_mask):
        tgt_mask = self.make_tgt_mask(tgt)

        tgt_embed = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        tgt_embed = self.pos_encoder(tgt_embed)

        dec_output = tgt_embed
        for layer in self.decoder:
            dec_output = layer(dec_output, encoder_output, tgt_mask, src_mask)
        return dec_output

    def forward(self, src, tgt):
        encoder_output, src_mask = self.encode(src)
        decoder_output = self.decode(tgt, encoder_output, src_mask)
        output = self.linear(decoder_output)
        return output

# 自检代码已迁移至 tests/test_model.py（运行: python -m pytest tests -v）
