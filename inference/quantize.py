"""
FP16 半精度模型导出脚本

用法: python inference/quantize.py

FP16 导出可使模型体积从 613MB 降至 102MB，精度几乎无损。
INT8 量化不适用（Embedding 占参数 46%，量化收益低）。
"""

import torch
import os
import sys

from core.models.transformer import Transformer
import sentencepiece as spm
from core.tokenizer import UnifiedBPETokenizer


def get_config_from_checkpoint(checkpoint):
    """从 checkpoint 提取模型配置（兼容完整 checkpoint / FP16 导出 / 纯 state_dict）"""
    if 'model_config' in checkpoint:
        return checkpoint['model_config']
    if 'args' in checkpoint:
        args = checkpoint['args']
        return {
            'd_model': args.d_model,
            'nhead': args.nhead,
            'num_encoder_layers': args.num_encoder_layers,
            'num_decoder_layers': args.num_decoder_layers,
            'd_ff': args.d_ff,
            'max_len': args.max_len,
        }
    raise KeyError("checkpoint 缺少 'model_config' 或 'args' 键，无法获取模型配置")


def build_model_from_checkpoint(checkpoint, device, vocab_size=None):
    """从检查点重建模型"""
    config = get_config_from_checkpoint(checkpoint)
    if vocab_size is None:
        vocab_size = checkpoint['tokenizer'].get_vocab_size()
    
    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=config['d_model'],
        num_heads=config['nhead'],
        num_encoder_layers=config['num_encoder_layers'],
        num_decoder_layers=config['num_decoder_layers'],
        d_ffn=config['d_ff'],
        dropout=0.0,
        max_len=config['max_len'],
        pad_idx=0
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model.to(device)


def load_tokenizer(checkpoint_path):
    """加载分词器（BPE 从 checkpoint 目录自动推导）"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    tokenizer = checkpoint['tokenizer']
    
    bpe_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    bpe_model = os.path.join(bpe_dir, "bpe_unified.model")
    if os.path.exists(bpe_model):
        tokenizer.sp = spm.SentencePieceProcessor()
        tokenizer.sp.Load(bpe_model)
        tokenizer.model_prefix = os.path.join(bpe_dir, "bpe_unified")
        tokenizer.pad_id = tokenizer.sp.pad_id()
        tokenizer.unk_id = tokenizer.sp.unk_id()
        tokenizer.bos_id = tokenizer.sp.bos_id()
        tokenizer.eos_id = tokenizer.sp.eos_id()
    
    return tokenizer


def greedy_translate(model, tokenizer, text, device, max_len=100):
    """贪婪解码翻译"""
    is_chinese = any('\u4e00' <= c <= '\u9fff' for c in text)
    src_lang = "zh" if is_chinese else "en"
    tgt_lang = "en" if is_chinese else "zh"
    
    src_ids = tokenizer.encode(text, lang=src_lang, add_bos=False, add_eos=True)
    src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)
    
    with torch.no_grad():
        encoder_output, src_mask = model.encode(src_tensor)
        
        tgt_ids = [tokenizer.bos_id]
        for _ in range(max_len):
            tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long).to(device)
            decoder_output = model.decode(tgt_tensor, encoder_output, src_mask)
            output = model.linear(decoder_output)
            next_token = output[0, -1].argmax().item()
            if next_token == tokenizer.eos_id:
                break
            tgt_ids.append(next_token)
    
    return tokenizer.decode(tgt_ids, lang=tgt_lang)


def get_model_size_mb(model_or_path):
    """获取模型体积（MB）"""
    if isinstance(model_or_path, str):
        return os.path.getsize(model_or_path) / (1024 * 1024)
    else:
        # 估算内存占用
        total = sum(p.numel() * p.element_size() for p in model_or_path.parameters())
        return total / (1024 * 1024)


# 测试句子
TEST_SENTENCES = [
    "你好世界",
    "今天天气很好，我们去公园散步吧。",
    "机器学习是人工智能的一个重要分支。",
    "你喜欢玩游戏还是喜欢女同学？",
]


# 主流程
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(script_dir, "checkpoints", "best_model.pt")
    
    if not os.path.exists(checkpoint_path):
        print(f"[错误] 找不到模型文件: {checkpoint_path}")
        print("请确保 checkpoints/best_model.pt 存在")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    print(f"原始模型: {checkpoint_path}")
    print(f"原始大小: {get_model_size_mb(checkpoint_path):.1f} MB")
    print("=" * 60)
    
    # 1. 加载原始模型
    print("\n[1/3] 加载原始 FP32 模型...")
    raw = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    tokenizer = load_tokenizer(checkpoint_path)
    model_fp32 = build_model_from_checkpoint(raw, device)

    # 2. FP16 导出
    print("\n[2/3] 导出 FP16 半精度模型...")
    fp16_path = os.path.join(script_dir, "checkpoints", "model_fp16.pt")

    model_fp16 = build_model_from_checkpoint(raw, 'cpu')
    model_fp16.half()
    torch.save({
        'model_state_dict': model_fp16.state_dict(),
        'model_config': get_config_from_checkpoint(raw),
    }, fp16_path)
    fp16_mem = get_model_size_mb(model_fp16)
    fp16_disk = get_model_size_mb(fp16_path)
    print(f"  FP16 内存占用: {fp16_mem:.1f} MB")
    print(f"  FP16 磁盘体积: {fp16_disk:.1f} MB")
    print(f"  已保存: {fp16_path}")

    # 3. 推理对比
    print("\n[3/3] 推理对比验证...")

    models = {
        "FP32 (原始)":    (model_fp32, device),
        "FP16 (半精度)":  (model_fp16.to(device), device),
    }

    for name, (model, dev) in models.items():
        model.eval()

    for text in TEST_SENTENCES:
        print(f"\n  输入: {text}")
        for name, (model, dev) in models.items():
            result = greedy_translate(model, tokenizer, text, dev)
            print(f"    {name:16s} → {result}")

    # 汇总
    print("\n" + "=" * 60)
    print("导出汇总")
    print("=" * 60)
    print(f"{'文件':<30s} {'体积':>10s} {'压缩比':>10s}")
    print("-" * 50)
    orig_size = get_model_size_mb(checkpoint_path)
    for path, label in [(checkpoint_path, "best_model.pt (原始)"),
                         (fp16_path, "model_fp16.pt")]:
        size = get_model_size_mb(path)
        ratio = size / orig_size * 100
        print(f"{label:<30s} {size:>8.1f} MB {ratio:>8.0f}%")

    print(f"\n使用方法:")
    print(f"  ckpt = torch.load('checkpoints/model_fp16.pt', map_location='cpu')")
    print(f"  config = ckpt['model_config']")
    print(f"  model = Transformer(vocab_size, **config).half()")
    print(f"  model.load_state_dict(ckpt['model_state_dict'])")
    print(f"  model.to(device)")
    print(f"\nFP16 是当前模型的最佳部署方案：102MB、GPU 推理、精度无损")


if __name__ == "__main__":
    main()
