"""
数据预处理流程脚本

用法: python tools/preprocess_pipeline.py
"""

import os
import random


def step1_process_csv(input_csv, output_dir):
    """
    步骤1: 处理原始CSV，生成清洗后的平行语料
    
    调用: python tools/process_wmt.py --input <csv> --output_dir <dir>
    """
    print("="*60)
    print("步骤1: 清洗原始CSV数据")
    print("="*60)
    cmd = f"python tools/process_wmt.py --input {input_csv} --output_dir {output_dir}"
    print(f"执行: {cmd}")
    os.system(cmd)
    print()


def step2_sample_data(full_zh, full_en, train_num=1000000, valid_num=100000, seed=42):
    """
    步骤2: 从全量数据中随机采样训练集和验证集
    
    参数:
        full_zh: 全量中文文件路径
        full_en: 全量英文文件路径
        train_num: 训练集句对数 (默认100万)
        valid_num: 验证集句对数 (默认10万)
        seed: 随机种子
    """
    print("="*60)
    print(f"步骤2: 采样数据 (训练集{train_num}句, 验证集{valid_num}句)")
    print("="*60)
    
    # 读取全量数据
    with open(full_zh, 'r', encoding='utf-8') as f:
        zh_lines = f.readlines()
    with open(full_en, 'r', encoding='utf-8') as f:
        en_lines = f.readlines()
    
    total = len(zh_lines)
    print(f"全量数据: {total} 句对")
    
    # 随机采样训练集
    random.seed(seed)
    train_num = min(train_num, total)
    train_indices = random.sample(range(total), train_num)
    train_set = set(train_indices)
    
    # 从剩余数据中采样验证集
    remaining = [i for i in range(total) if i not in train_set]
    random.seed(seed + 1)
    valid_num = min(valid_num, len(remaining))
    valid_indices = random.sample(remaining, valid_num) if valid_num > 0 else []
    
    output_dir = os.path.dirname(full_zh)
    
    # 写入训练集
    with open(os.path.join(output_dir, 'train.zh'), 'w', encoding='utf-8') as f:
        for i in train_indices:
            f.write(zh_lines[i])
    with open(os.path.join(output_dir, 'train.en'), 'w', encoding='utf-8') as f:
        for i in train_indices:
            f.write(en_lines[i])
    
    # 写入验证集
    with open(os.path.join(output_dir, 'valid.zh'), 'w', encoding='utf-8') as f:
        for i in valid_indices:
            f.write(zh_lines[i])
    with open(os.path.join(output_dir, 'valid.en'), 'w', encoding='utf-8') as f:
        for i in valid_indices:
            f.write(en_lines[i])
    
    print(f"训练集: {train_num} 句对 → train.zh/en")
    print(f"验证集: {valid_num} 句对 → valid.zh/en")
    print()


def step3_train_tokenizer(zh_file, en_file, vocab_size=32000, output_dir="./checkpoints"):
    """
    步骤3: 训练BPE分词器
    
    调用: python tools/train_tokenizer.py
    或直接使用 core.tokenizer 中的 build_tokenizer
    """
    print("="*60)
    print("步骤3: 训练BPE分词器")
    print("="*60)
    
    import sentencepiece as spm
    import tempfile
    import re
    
    os.makedirs(output_dir, exist_ok=True)
    output_prefix = os.path.join(output_dir, "bpe_unified")
    
    if os.path.exists(output_prefix + ".model"):
        print(f"BPE模型已存在: {output_prefix}")
        print("如需重新训练，请删除现有模型文件")
        return
    
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8')
    
    try:
        with open(zh_file, 'r', encoding='utf-8') as f:
            for line in f:
                text = line.strip()
                if text:
                    temp_file.write("▁zh " + text + "\n")
        
        with open(en_file, 'r', encoding='utf-8') as f:
            for line in f:
                text = line.strip().lower()
                if text:
                    text = re.sub(r'([.,!?;:])', r' \1', text)
                    temp_file.write("▁en " + text + "\n")
        
        temp_file.close()
        
        print(f"训练BPE模型，词表大小: {vocab_size}")
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
            input_sentence_size=1000000,
            shuffle_input_sentence=True,
        )
        print(f"BPE分词器训练完成: {output_prefix}.model")
        
    finally:
        os.unlink(temp_file.name)
    
    print()


def run_full_pipeline():
    """
    执行完整预处理流程
    """
    print("\n" + "="*60)
    print("Transformer 中英翻译 - 数据预处理流程")
    print("="*60 + "\n")
    
    # 配置路径
    raw_csv = "data/WMT-CN-to-EN/wmt_zh_en_training_corpus.csv"
    processed_dir = "data/wmt_processed"
    
    # 步骤1: 清洗CSV (如果输出文件不存在)
    full_zh = os.path.join(processed_dir, "wmt_zh_en_training_corpus.zh")
    full_en = os.path.join(processed_dir, "wmt_zh_en_training_corpus.en")
    
    if not os.path.exists(full_zh) or not os.path.exists(full_en):
        step1_process_csv(raw_csv, processed_dir)
    else:
        print("步骤1: 清洗文件已存在，跳过")
        print()
    
    # 步骤2: 采样 (如果train文件不存在)
    train_zh = os.path.join(processed_dir, "train.zh")
    if not os.path.exists(train_zh):
        step2_sample_data(full_zh, full_en, train_num=1000000, valid_num=100000)
    else:
        print("步骤2: 采样文件已存在，跳过")
        print()
    
    # 步骤3: 训练分词器
    step3_train_tokenizer(
        os.path.join(processed_dir, "train.zh"),
        os.path.join(processed_dir, "train.en"),
        vocab_size=32000,
        output_dir="./checkpoints"
    )
    
    print("="*60)
    print("预处理完成！")
    print("="*60)


if __name__ == "__main__":
    run_full_pipeline()
