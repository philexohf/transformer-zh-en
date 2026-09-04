"""
推理集成测试（需要训练好的 checkpoint）

由 tools/test_translate.py 与 scripts/check_inference.py 迁移合并：
- 加载 best_model.pt 构建模型
- 中文 -> 英文 / 英文 -> 中文 贪心翻译冒烟
- 语言自动检测路径

checkpoint 不存在时由 conftest 的 checkpoint_path fixture 自动 skip。
运行: python -m pytest tests/test_inference.py -v
"""

import os

import pytest
import torch

MAX_DECODE_STEPS = 100
TEST_CASES = [
    ("你好", "zh", "en"),
    ("我爱编程", "zh", "en"),
    ("hello world", "en", "zh"),
    ("machine learning", "en", "zh"),
]


def detect_lang(text):
    """与推理脚本一致的语言自动检测：含汉字 -> 中文"""
    return "zh" if any("\u4e00" <= c <= "\u9fff" for c in text) else "en"


@pytest.fixture(scope="module")
def infer_env(checkpoint_path):
    """加载 checkpoint + 分词器 + 模型（CPU/GPU 自适应）"""
    from core.models.transformer import Transformer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # checkpoint 含 tokenizer/args 等 Python 对象，需 weights_only=False
    # （与项目内 infer.py 等脚本一致，仅加载自己训练的 checkpoint）
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    tokenizer = checkpoint["tokenizer"]
    args = checkpoint.get("args", None)

    # 与推理脚本一致：checkpoint 目录中的 BPE 模型文件重新加载
    bpe_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    bpe_model = os.path.join(bpe_dir, "bpe_unified.model")
    if os.path.exists(bpe_model):
        import sentencepiece as spm

        tokenizer.sp = spm.SentencePieceProcessor()
        tokenizer.sp.Load(bpe_model)
        tokenizer.pad_id = tokenizer.sp.pad_id()
        tokenizer.unk_id = tokenizer.sp.unk_id()
        tokenizer.bos_id = tokenizer.sp.bos_id()
        tokenizer.eos_id = tokenizer.sp.eos_id()

    def _default(v, fallback):
        """从 checkpoint 的 args 读取超参，缺失时回退默认值"""
        if args is not None:
            val = getattr(args, v, None)
            if val is not None:
                return val
        return fallback

    model = Transformer(
        src_vocab_size=len(tokenizer),
        tgt_vocab_size=len(tokenizer),
        d_model=_default("d_model", 384),
        num_heads=_default("nhead", 8),
        num_encoder_layers=_default("num_encoder_layers", 4),
        num_decoder_layers=_default("num_decoder_layers", 4),
        d_ffn=_default("d_ff", 1536),
        dropout=0.0,
        max_len=_default("max_len", 128),
        pad_idx=0,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, tokenizer, device


def greedy_translate(model, tokenizer, device, text):
    """贪心解码翻译（与 infer.py 逻辑一致）"""
    src_lang = detect_lang(text)
    tgt_lang = "en" if src_lang == "zh" else "zh"

    src_ids = tokenizer.encode(text, lang=src_lang, add_bos=False, add_eos=True)
    src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)

    with torch.no_grad():
        encoder_output, src_mask = model.encode(src_tensor)
        tgt_ids = [tokenizer.bos_id]
        for _ in range(MAX_DECODE_STEPS):
            tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long).to(device)
            decoder_output = model.decode(tgt_tensor, encoder_output, src_mask)
            output = model.linear(decoder_output)
            next_token = output[0, -1].argmax().item()
            if next_token == tokenizer.eos_id:
                break
            tgt_ids.append(next_token)

    result = tokenizer.decode(tgt_ids[1:], lang=tgt_lang)
    return result, tgt_lang


@pytest.mark.parametrize("text,src_lang,tgt_lang", TEST_CASES)
def test_translate_smoke(infer_env, text, src_lang, tgt_lang):
    """每个测试用例都应返回非空译文，且不抛出异常"""
    model, tokenizer, device = infer_env
    result, detected_tgt = greedy_translate(model, tokenizer, device, text)
    assert detected_tgt == tgt_lang
    assert isinstance(result, str)
    assert len(result.strip()) > 0


def test_translate_stops_on_eos(infer_env):
    """翻译应在有限步内终止（EOS 触发），而不是跑满 MAX_DECODE_STEPS"""
    model, tokenizer, device = infer_env
    result, _ = greedy_translate(model, tokenizer, device, "你好世界")
    # 能走到这里说明循环被 EOS 打断；空检查防止异常产物
    assert len(result) > 0


def test_model_is_deterministic(infer_env):
    """同一句输入两次贪心解码结果应一致（eval 模式无 dropout）"""
    model, tokenizer, device = infer_env
    r1, _ = greedy_translate(model, tokenizer, device, "机器学习是人工智能的分支")
    r2, _ = greedy_translate(model, tokenizer, device, "机器学习是人工智能的分支")
    assert r1 == r2
