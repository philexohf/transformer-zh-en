"""
BPE分词器训练脚本

功能：训练统一的中英文 BPE 分词器，中英文共用一个词表
使用：python train_tokenizer.py
输出：checkpoints/bpe_unified.model, checkpoints/bpe_unified.vocab
"""

import os
import sentencepiece as spm
import tempfile
import re


def train_bpe_tokenizer(zh_file, en_file, vocab_size=32000, output_dir="./checkpoints"):
    """
    训练 BPE 分词器
    
    参数:
        zh_file: 中文训练数据文件路径
        en_file: 英文训练数据文件路径
        vocab_size: 词表大小（默认32000）
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    output_prefix = os.path.join(output_dir, "bpe_unified")
    
    # 检查是否已有模型
    if os.path.exists(output_prefix + ".model"):
        print(f"BPE 模型已存在: {output_prefix}")
        print("如需重新训练，请删除现有模型文件")
        return output_prefix
    
    print("=" * 50)
    print("开始训练 BPE 分词器")
    print("=" * 50)
    
    # 创建临时合并文件
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8')
    
    try:
        # 处理中文
        print(f"处理中文数据: {zh_file}")
        with open(zh_file, 'r', encoding='utf-8') as f:
            for line in f:
                text = line.strip()
                if text:
                    # ▁zh 是 SentencePiece 的句子前缀标记
                    temp_file.write("▁zh " + text + "\n")
        
        # 处理英文
        print(f"处理英文数据: {en_file}")
        with open(en_file, 'r', encoding='utf-8') as f:
            for line in f:
                text = line.strip().lower()
                if text:
                    # 处理标点符号
                    text = re.sub(r'([.,!?;:])', r' \1', text)
                    temp_file.write("▁en " + text + "\n")
        
        temp_file.close()
        
        # 训练 BPE 模型
        print(f"训练 BPE 模型，词表大小: {vocab_size}")
        print("限制训练数据为 100 万条以加快速度...")
        spm.SentencePieceTrainer.train(
            input=temp_file.name,
            vocab_size=vocab_size,
            model_prefix=output_prefix,
            character_coverage=1.0,
            model_type='bpe',
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            normalization_rule_name='nmt_nfkc',
            user_defined_symbols='▁zh,▁en',
            input_sentence_size=1000000,  # 限制为100万条
            shuffle_input_sentence=True,
        )
        
        print("=" * 50)
        print(f"BPE 分词器训练完成！")
        print(f"模型文件: {output_prefix}.model")
        print(f"词表文件: {output_prefix}.vocab")
        print("=" * 50)
        
    finally:
        os.unlink(temp_file.name)
    
    return output_prefix


def main():
    # 数据路径 - 247万subset数据
    zh_file = "./data/wmt_processed/subset.zh"
    en_file = "./data/wmt_processed/subset.en"
    
    # 训练分词器
    train_bpe_tokenizer(
        zh_file=zh_file,
        en_file=en_file,
        vocab_size=32000,
        output_dir="./checkpoints"
    )
    
    # 测试分词器
    print("\n测试分词器:")
    sp = spm.SentencePieceProcessor()
    sp.Load("./checkpoints/bpe_unified.model")
    
    # 测试中文
    zh_text = "机器翻译是人工智能的重要应用"
    zh_ids = sp.Encode(zh_text, add_bos=True, add_eos=True)
    zh_pieces = [sp.id_to_piece(i) for i in zh_ids]
    print(f"中文: {zh_text}")
    print(f"  IDs: {zh_ids}")
    print(f"  Pieces: {zh_pieces}")
    
    # 测试英文
    en_text = "machine translation is an important application of artificial intelligence"
    en_ids = sp.Encode(en_text, add_bos=True, add_eos=True)
    en_pieces = [sp.id_to_piece(i) for i in en_ids]
    print(f"\n英文: {en_text}")
    print(f"  IDs: {en_ids}")
    print(f"  Pieces: {en_pieces}")


if __name__ == "__main__":
    main()