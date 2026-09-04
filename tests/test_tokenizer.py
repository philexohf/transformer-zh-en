"""
分词器测试

由 tools/check_bpe.py 与 scripts/check_data.py 中的断言部分迁移而来：
- 词表加载与特殊 token id
- 中英文 encode/decode 往返一致性
- piece <-> id 映射一致性

运行: python -m pytest tests/test_tokenizer.py -v
"""

import pytest

# BPE 模型缺失时 conftest 的 tokenizer fixture 会自动 skip


def _normalized(text):
    """往返比较用：忽略空格差异（SP 的 ▁ 词边界在中文场景会被移除）"""
    return text.replace(" ", "").lower()


def test_vocab_loaded(tokenizer):
    """词表应已加载且大于特殊 token 数"""
    assert tokenizer.get_vocab_size() > 10


def test_special_ids_are_sentencepiece_defaults(tokenizer):
    """SP 默认：pad=0, unk=1, bos=2, eos=3"""
    assert tokenizer.pad_id == 0
    assert tokenizer.unk_id == 1
    assert tokenizer.bos_id == 2
    assert tokenizer.eos_id == 3


def test_special_token_pieces(tokenizer):
    """特殊 token 的 piece 名称应与 id 对应（check_bpe.py 逻辑）"""
    for i, expected in [
        (tokenizer.pad_id, tokenizer.pad_token),
        (tokenizer.unk_id, tokenizer.unk_token),
        (tokenizer.bos_id, tokenizer.bos_token),
        (tokenizer.eos_id, tokenizer.eos_token),
    ]:
        assert tokenizer.sp.id_to_piece(i) == expected


def test_piece_id_roundtrip_consistency(tokenizer):
    """piece_to_id(id_to_piece(i)) == i（check_bpe.py 抽样检查）"""
    vocab = tokenizer.get_vocab_size()
    for i in [0, 1, 2, 3, 100, 1000, 3432, min(26211, vocab - 1)]:
        if i < vocab:
            piece = tokenizer.sp.id_to_piece(i)
            assert tokenizer.sp.piece_to_id(piece) == i


def test_zh_encode_decode_roundtrip(tokenizer):
    """中文往返：decode(encode(text)) 应还原原文（无空格差异）

    注意：nmt_nfkc 归一化会把全角标点折叠为 ASCII，往返断言避开标点。
    """
    text = "你好世界这是一个测试"
    zh_ids = tokenizer.encode(text, lang="zh", add_bos=False, add_eos=True)
    decoded = tokenizer.decode(zh_ids)
    assert _normalized(decoded) == _normalized(text)
    # 中文编码不应带 BOS，但应带 EOS
    assert zh_ids[0] != tokenizer.bos_id
    assert zh_ids[-1] == tokenizer.eos_id


def test_fullwidth_punct_normalized_to_ascii(tokenizer):
    """nmt_nfkc 归一化：全角逗号应等价于 ASCII 逗号（编码一致）

    注意：句号（U+3002）不是 NFKC 兼容字符，不会被折叠，勿混入测试。
    """
    ids_fullwidth = tokenizer.encode("你好，世界", lang="zh")
    ids_ascii = tokenizer.encode("你好,世界", lang="zh")
    assert ids_fullwidth == ids_ascii
    # 解码输出为归一化后的 ASCII 标点（原始全角无法还原）
    decoded = tokenizer.decode(ids_fullwidth)
    assert "，" not in decoded
    assert all(i != tokenizer.unk_id for i in ids_fullwidth)


def test_en_encode_decode_roundtrip(tokenizer):
    """英文往返：decode(encode(text)) 应还原原文（忽略空格、统一小写）"""
    text = "Hello world, this is a test."
    en_ids = tokenizer.encode(text, lang="en", add_bos=True, add_eos=True)
    decoded = tokenizer.decode(en_ids)
    assert _normalized(decoded) == _normalized(text)
    assert en_ids[0] == tokenizer.bos_id
    assert en_ids[-1] == tokenizer.eos_id


def test_bos_eos_flags_control_encoding(tokenizer):
    """add_bos / add_eos 开关应精确控制特殊 token 的插入"""
    ids_all = tokenizer.encode("你好", lang="zh", add_bos=True, add_eos=True)
    ids_none = tokenizer.encode("你好", lang="zh", add_bos=False, add_eos=False)
    assert ids_all[0] == tokenizer.bos_id
    assert ids_all[-1] == tokenizer.eos_id
    assert ids_none[0] != tokenizer.bos_id
    assert ids_none[-1] != tokenizer.eos_id
    # 去掉特殊 token 后核心内容一致
    assert ids_all[1:-1] == ids_none


def test_zh_en_share_vocab(tokenizer):
    """统一词表：中文与英文的 id 都落在同一 [0, vocab) 区间"""
    zh_ids = tokenizer.encode("机器翻译", lang="zh")
    en_ids = tokenizer.encode("machine translation", lang="en")
    vocab = tokenizer.get_vocab_size()
    assert all(0 <= i < vocab for i in zh_ids + en_ids)
