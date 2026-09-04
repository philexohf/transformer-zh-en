"""
数据集与 collate 测试

由 scripts/check_data.py 中的断言部分迁移而来，改用 tmp_path 生成小型
平行语料，避免依赖仓库内的大数据文件：
- 样本结构与特殊 token 约定（src 无 BOS 有 EOS；tgt 有 BOS+EOS）
- 超长截断后末尾必须是 EOS
- collate_fn 的 padding / 形状正确性
- 中英文语料不对齐时应报错

运行: python -m pytest tests/test_data.py -v
"""

import pytest
import torch

from dataset import TranslationDataset, collate_fn

ZH_LINES = [
    "你好世界",
    "机器学习是人工智能的一个重要分支。",
    "今天天气很好，我们去公园散步吧。",
]
EN_LINES = [
    "hello world",
    "machine learning is an important branch of artificial intelligence .",
    "the weather is nice today , let us go for a walk in the park .",
]

MAX_LEN = 20


@pytest.fixture
def corpus_dir(tmp_path):
    """构造微型平行语料目录（含一条超长句子用于截断测试）"""
    zh = ZH_LINES + ["这个句子特别长" * 20]  # 140 字，必然超 MAX_LEN
    en = EN_LINES + ["this sentence is very long " * 20]
    (tmp_path / "train.zh").write_text("\n".join(zh), encoding="utf-8")
    (tmp_path / "train.en").write_text("\n".join(en), encoding="utf-8")
    return tmp_path


def test_dataset_length(corpus_dir, tokenizer):
    ds = TranslationDataset(str(corpus_dir), tokenizer, MAX_LEN, "train")
    assert len(ds) == 4


def test_sample_special_token_convention(corpus_dir, tokenizer):
    """src：无 BOS、末尾 EOS；tgt：开头 BOS、末尾 EOS"""
    ds = TranslationDataset(str(corpus_dir), tokenizer, MAX_LEN, "train")
    sample = ds[0]
    src, tgt = sample["src"], sample["tgt"]
    assert src[0] != tokenizer.bos_id
    assert src[-1] == tokenizer.eos_id
    assert tgt[0] == tokenizer.bos_id
    assert tgt[-1] == tokenizer.eos_id


def test_truncation_ends_with_eos(corpus_dir, tokenizer):
    """超长序列截断后，最后一个位置必须是 EOS（模型停止信号不丢失）"""
    ds = TranslationDataset(str(corpus_dir), tokenizer, MAX_LEN, "train")
    sample = ds[-1]  # 超长样本
    assert len(sample["src"]) <= MAX_LEN
    assert sample["src"][-1] == tokenizer.eos_id
    assert len(sample["tgt"]) <= MAX_LEN
    assert sample["tgt"][-1] == tokenizer.eos_id


def test_collate_padding_shape_and_pad_id(corpus_dir, tokenizer):
    """collate_fn 应把 batch 内各样本 padding 到最长，pad 位置填 pad_id"""
    ds = TranslationDataset(str(corpus_dir), tokenizer, MAX_LEN, "train")
    indices = torch.randperm(len(ds))[:3].tolist()  # 随机取 3 条不同长度样本
    batch = [ds[i] for i in indices]
    src, tgt = collate_fn(batch)

    max_src = max(len(b["src"]) for b in batch)
    max_tgt = max(len(b["tgt"]) for b in batch)
    assert src.shape == (3, max_src)
    assert tgt.shape == (3, max_tgt)

    # 每个样本的有效区不含 pad；pad 区全部为 pad_id
    for b, s in zip(batch, src):
        assert s[len(b["src"]):].eq(tokenizer.pad_id).all()
    for b, t in zip(batch, tgt):
        assert t[len(b["tgt"]):].eq(tokenizer.pad_id).all()


def test_collate_batch_keeps_bos_eos(corpus_dir, tokenizer):
    """padding 后 BOS/EOS 约定不应被破坏"""
    ds = TranslationDataset(str(corpus_dir), tokenizer, MAX_LEN, "train")
    batch = [ds[i] for i in range(3)]
    src, tgt = collate_fn(batch)
    assert (tgt[:, 0] == tokenizer.bos_id).all()
    assert (tgt == tokenizer.eos_id).any(dim=1).all()
    assert (src == tokenizer.eos_id).any(dim=1).all()


def test_mismatched_parallel_corpus_raises(tmp_path, tokenizer):
    """中英文行数不对齐时必须报错，防止静默错位训练"""
    (tmp_path / "train.zh").write_text("\n".join(ZH_LINES), encoding="utf-8")
    (tmp_path / "train.en").write_text("\n".join(EN_LINES[:2]), encoding="utf-8")
    with pytest.raises(AssertionError, match="Data mismatch"):
        TranslationDataset(str(tmp_path), tokenizer, MAX_LEN, "train")
