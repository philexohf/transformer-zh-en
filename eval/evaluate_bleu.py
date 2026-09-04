"""
BLEU 评估脚本

计算翻译质量（标准 BLEU 公式）。
用法: python eval/evaluate_bleu.py --checkpoint ./checkpoints/best_model.pt
"""

import os
# Force CUDA device visibility early so torch picks up GPU on Windows
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
import torch
import argparse
from tqdm import tqdm
import sacrebleu

from core.config import get_args
from core.transformer import Transformer
from core.tokenizer import build_tokenizer
from core.dataset import TranslationDataset, collate_fn
from torch.utils.data import DataLoader


def compute_bleu(predictions, references):
    """
    Standard BLEU score using sacrebleu (4-gram precision + brevity penalty).
    Returns BLEU score on 0-100 scale.
    """
    if len(predictions) == 0:
        return 0.0
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    return bleu.score


def load_checkpoint(checkpoint_path, device, vocab_size):
    """加载模型和分词器"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    tokenizer = checkpoint['tokenizer']
    args = checkpoint.get('args', None)
    if args is None:
        from core.config import Config
        args = Config()
    
    # 从 checkpoint 路径推导 BPE 目录
    bpe_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    bpe_model = os.path.join(bpe_dir, "bpe_unified.model")
    
    # 重新加载 BPE
    import sentencepiece as spm
    if os.path.exists(bpe_model):
        tokenizer.sp = spm.SentencePieceProcessor()
        tokenizer.sp.Load(bpe_model)
        tokenizer.pad_id = tokenizer.sp.pad_id()
        tokenizer.unk_id = tokenizer.sp.unk_id()
        tokenizer.bos_id = tokenizer.sp.bos_id()
        tokenizer.eos_id = tokenizer.sp.eos_id()
    
    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=args.d_model,
        num_heads=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        d_ffn=args.d_ff,
        dropout=args.dropout,
        max_len=args.max_len,
        pad_idx=0
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, tokenizer, args


def translate(model, tokenizer, src_ids, device, max_len=100):
    """翻译单个句子"""
    src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)
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
    
    return tokenizer.decode(tgt_ids[1:], lang="en")  # 跳过 BOS


def evaluate_bleu(model, tokenizer, val_dataset, device, max_len=100, batch_size=32):
    """计算 BLEU 分数"""
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    predictions = []
    references = []
    
    print("Evaluating...")
    for batch in tqdm(val_loader):
        src, tgt = batch
        src = src.to(device)
        
        # 翻译
        for i in range(src.size(0)):
            src_ids = src[i].tolist()
            # 移除 padding
            src_ids = [x for x in src_ids if x != tokenizer.pad_id]
            
            pred = translate(model, tokenizer, src_ids, device, max_len)
            predictions.append(pred)
            
            # 获取参考译文
            tgt_ids = tgt[i].tolist()
            tgt_ids = [x for x in tgt_ids if x not in [tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id]]
            ref = tokenizer.decode(tgt_ids, lang="en")
            references.append(ref)
    
    # 计算 BLEU
    bleu = compute_bleu(predictions, references)
    return bleu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_model.pt")
    parser.add_argument("--max_len", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=1000, help="最多评估多少样本")
    args = parser.parse_args()
    
    # 从 checkpoint 路径推导 BPE 目录
    bpe_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载分词器
    tokenizer = build_tokenizer(
        "./data/wmt_processed/train.zh",
        "./data/wmt_processed/train.en",
        32000,
        os.path.join(bpe_dir, "bpe_unified")
    )
    vocab_size = len(tokenizer)
    
    print(f"Loading checkpoint: {args.checkpoint}")
    model, tokenizer, train_args = load_checkpoint(args.checkpoint, device, vocab_size)
    
    val_dataset = TranslationDataset("./data/wmt_processed", tokenizer, args.max_len, "valid")
    
    # 限制评估样本数
    if args.max_samples < len(val_dataset):
        val_dataset.zh_lines = val_dataset.zh_lines[:args.max_samples]
        val_dataset.en_lines = val_dataset.en_lines[:args.max_samples]
    
    print(f"Evaluating {len(val_dataset)} samples...")
    
    # 计算 BLEU
    bleu_score = evaluate_bleu(model, tokenizer, val_dataset, device, args.max_len, args.batch_size)
    
    print(f"\n{'='*50}")
    print(f"BLEU Score: {bleu_score:.2f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()