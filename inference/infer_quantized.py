"""
FP16 半精度模型推理脚本

用法:
  python inference/infer_quantized.py                    # 交互式
  python inference/infer_quantized.py --input "你好世界"  # 单句翻译

FP16: GPU 推理，速度最快，精度几乎无损，体积仅 102MB
"""

import torch
import os
import argparse

from core.transformer import Transformer
from core.tokenizer import UnifiedBPETokenizer


def build_model(vocab_size, config):
    """构建与训练时结构一致的 Transformer 模型"""
    return Transformer(
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


def load_fp16_model(checkpoint_dir, tokenizer, device):
    """
    加载 FP16 半精度模型

    从 model_fp16.pt 中读取模型配置和权重，不再硬编码 Config。
    """
    vocab_size = tokenizer.get_vocab_size()

    fp16_path = os.path.join(checkpoint_dir, "model_fp16.pt")
    if not os.path.exists(fp16_path):
        raise FileNotFoundError(
            f"找不到 FP16 模型: {fp16_path}\n"
            f"请先运行 python quantize.py 导出"
        )

    checkpoint = torch.load(fp16_path, map_location='cpu', weights_only=True)
    config = checkpoint['model_config']

    model = build_model(vocab_size, config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.half()
    model.eval()

    print(f"  FP16 模型已加载: {fp16_path}")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
    return model.to(device), config


def load_tokenizer(checkpoint_dir):
    """加载分词器（直接加载 BPE 模型，无需 best_model.pt）"""
    bpe_prefix = os.path.join(checkpoint_dir, "bpe_unified")
    tokenizer = UnifiedBPETokenizer(bpe_prefix)
    
    if tokenizer.sp is None:
        raise FileNotFoundError(
            f"找不到 BPE 模型: {bpe_prefix}.model\n"
            f"请将 bpe_unified.model 和 bpe_unified.vocab 放在本目录"
        )
    
    tokenizer.pad_id = tokenizer.sp.pad_id()
    tokenizer.unk_id = tokenizer.sp.unk_id()
    tokenizer.bos_id = tokenizer.sp.bos_id()
    tokenizer.eos_id = tokenizer.sp.eos_id()
    
    return tokenizer


def translate(model, tokenizer, text, config):
    """
    翻译函数（贪婪解码）

    自动检测输入语言：中文->英文，英文->中文
    """
    is_chinese = any('\u4e00' <= c <= '\u9fff' for c in text)
    src_lang = "zh" if is_chinese else "en"
    tgt_lang = "en" if is_chinese else "zh"

    # 自动适配模型所在设备（INT8 在 CPU 上）
    model_device = next(model.parameters()).device

    src_ids = tokenizer.encode(text, lang=src_lang, add_bos=False, add_eos=True)
    src_tensor = torch.tensor([src_ids], dtype=torch.long).to(model_device)

    with torch.no_grad():
        encoder_output, src_mask = model.encode(src_tensor)

        tgt_ids = [tokenizer.bos_id]
        for _ in range(config['max_len']):
            tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long).to(model_device)
            decoder_output = model.decode(tgt_tensor, encoder_output, src_mask)
            output = model.linear(decoder_output)
            next_token = output[0, -1].argmax().item()
            if next_token == tokenizer.eos_id:
                break
            tgt_ids.append(next_token)

    return tokenizer.decode(tgt_ids, lang=tgt_lang)


def main():
    parser = argparse.ArgumentParser(description="FP16 半精度模型推理")
    parser.add_argument(
        "--input", type=str, default=None,
        help="单句翻译（不指定则进入交互模式）"
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(script_dir, "checkpoints")

    print("=" * 50)
    print("加载 FP16 模型")
    print("=" * 50)
    tokenizer = load_tokenizer(checkpoint_dir)
    print(f"  词表大小: {tokenizer.get_vocab_size()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("  [警告] 未检测到 GPU，回退到 CPU（速度较慢）")
    model, config = load_fp16_model(checkpoint_dir, tokenizer, device)
    print(f"  运行设备: {device}")

    if args.input:
        result = translate(model, tokenizer, args.input, config)
        print(f"\nSource:      {args.input}")
        print(f"Translation: {result}")
    else:
        print("\n" + "=" * 50)
        print("交互式翻译 (FP16)")
        print("输入文本后回车，输入 'quit' 退出")
        print("=" * 50)

        while True:
            try:
                text = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见！")
                break

            if text.lower() == "quit":
                print("再见！")
                break
            if not text:
                continue

            result = translate(model, tokenizer, text, config)
            print(f"  {result}")


if __name__ == "__main__":
    main()
