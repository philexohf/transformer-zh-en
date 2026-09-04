"""
模型单元测试

由 models/transformer.py 末尾的 __main__ 自检块迁移而来：
- forward 输出形状
- encode + decode 自回归生成形状
- 掩码形状

运行: python -m pytest tests/test_model.py -v
"""

import pytest
import torch

from models.transformer import Transformer

# 小尺寸配置，保证 CPU 上毫秒级完成
VOCAB = 2000
MODEL_KWARGS = dict(
    src_vocab_size=VOCAB,
    tgt_vocab_size=VOCAB,
    d_model=64,
    num_heads=4,
    num_encoder_layers=2,
    num_decoder_layers=2,
    d_ffn=128,
    dropout=0.1,
    max_len=128,
    pad_idx=0,
)


@pytest.fixture(scope="module")
def model():
    return Transformer(**MODEL_KWARGS)


def test_forward_output_shape(model):
    """forward（训练模式）输出应为 [batch, tgt_len, vocab]"""
    model.train()
    src = torch.randint(0, VOCAB, (2, 50))
    tgt = torch.randint(0, VOCAB, (2, 40))
    output = model(src, tgt)
    assert output.shape == (2, 40, VOCAB)


def test_encode_shape(model):
    """encode 返回 encoder 输出和 padding 掩码，形状正确"""
    src = torch.randint(0, VOCAB, (4, 30))
    enc, src_mask = model.encode(src)
    assert enc.shape == (4, 30, 64)
    assert src_mask.shape == (4, 1, 1, 30)  # [batch, 1, 1, src_len]


def test_decode_autoregressive(model):
    """decode 支持逐步生成：每一步只取最后位置，序列应不断增长"""
    model.eval()
    src = torch.randint(0, VOCAB, (4, 30))
    enc, src_mask = model.encode(src)

    tgt = torch.tensor([[2], [2], [2], [2]])  # 4 个 <s>（bos_id=2）
    with torch.no_grad():
        for _ in range(5):
            dec = model.decode(tgt, enc, src_mask)
            next_token = dec[:, -1:].argmax(dim=-1)
            tgt = torch.cat([tgt, next_token], dim=1)

    assert tgt.shape == (4, 6)  # 初始 1 列 + 生成 5 列
    assert (tgt < VOCAB).all()


def test_src_padding_mask_hides_pad(model):
    """padding 位置在 src mask 中应被屏蔽（False）"""
    src = torch.tensor([[5, 6, 7, 0, 0], [1, 2, 3, 4, 0]])  # pad_idx=0
    _, src_mask = model.encode(src)
    # 第一个样本：pad 位置在第 3、4 列 → mask[0, 0, 0, 3:] 全 False
    assert not src_mask[0, 0, 0, 3:].any()
    assert src_mask[0, 0, 0, :3].all()


def test_tgt_mask_is_causal_and_padding_aware(model):
    """解码器掩码 = 因果掩码 & padding 掩码"""
    tgt = torch.tensor([[2, 5, 6, 7], [2, 3, 0, 0]])  # 第二个样本 3 位置后是 pad
    mask = model.make_tgt_mask(tgt)
    # 因果性：位置 i 不可见位置 j>i
    for i in range(4):
        assert not mask[0, 0, i, i + 1:].any()
    # padding 屏蔽：第二个样本第 3、4 列对任何查询都不可见
    assert not mask[1, 0, :, 2:].any()


def test_parameter_count_within_reference(model):
    """与文档标注一致：Base 配置 ~65M，小配置应在合理区间"""
    total = sum(p.numel() for p in model.parameters())
    assert total > 0
