"""
Transformer 中英机器翻译 — 分词器
统一 BPE 分词，中英文使用相同子词词表。
"""

import re
import sentencepiece as spm
import os
import tempfile


class UnifiedBPETokenizer:
    """
    统一 BPE 分词器

    中英文使用相同 BPE 词表（32K），粒度一致、共享表示、解决 OOV。
    """
    
    def __init__(self, model_prefix=None):
        self.sp = None
        self.model_prefix = model_prefix
        
        # SentencePiece 默认特殊 token
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3
        
        if model_prefix and os.path.exists(model_prefix + ".model"):
            self.sp = spm.SentencePieceProcessor()
            self.sp.Load(model_prefix + ".model")
    
    def train(self, zh_file, en_file, vocab_size=32000, model_prefix="bpe_unified"):
        """
        训练 BPE 分词模型。
        合并中英文语料并添加语言标记，用 SentencePiece 训练后保存。
        """
        self.model_prefix = model_prefix
        
        # 已有模型直接加载
        if os.path.exists(model_prefix + ".model"):
            self.sp = spm.SentencePieceProcessor()
            self.sp.Load(model_prefix + ".model")
            print(f"Loaded existing BPE model: {model_prefix}")
            return
        
        print("Training unified BPE tokenizer...")
        
        # 创建临时合并文件
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8')
        
        try:
            # 处理中文：添加语言标记前缀
            with open(zh_file, 'r', encoding='utf-8') as f:
                for line in f:
                    text = line.strip()
                    if text:
                        # ▁zh 是SentencePiece的句子前缀标记
                        temp_file.write("▁zh " + text + "\n")
            
            # 处理英文：转小写 + 标点处理 + 语言标记
            with open(en_file, 'r', encoding='utf-8') as f:
                for line in f:
                    text = line.strip().lower()
                    if text:
                        text = re.sub(r'([.,!?;:])', r' \1', text)
                        temp_file.write("▁en " + text + "\n")
            
            temp_file.close()
            
            # 训练BPE模型
            spm.SentencePieceTrainer.train(
                input=temp_file.name,
                vocab_size=vocab_size,
                model_prefix=model_prefix,
                character_coverage=1.0,
                model_type='bpe',
                pad_id=self.pad_id,
                unk_id=self.unk_id,
                bos_id=self.bos_id,
                eos_id=self.eos_id,
                normalization_rule_name='nmt_nfkc',
                user_defined_symbols='▁zh,▁en',
                input_sentence_size=1000000,
                shuffle_input_sentence=True,
            )
            
            # 加载训练好的模型
            self.sp = spm.SentencePieceProcessor()
            self.sp.Load(model_prefix + ".model")
            
            print(f"BPE model trained: vocab_size={vocab_size}")
            
        finally:
            os.unlink(temp_file.name)
    
    def tokenize_zh(self, text):
        """中文分词，加语言标记前缀后调用 SentencePiece。"""
        if self.sp is None:
            raise ValueError("BPE model not loaded")
        
        text = "▁zh " + text
        pieces = self.sp.encode(text, out_type=str)
        
        # 过滤语言标记
        result = [p for p in pieces if p not in ['▁zh', '▁en']]
        return result
    
    def tokenize_en(self, text):
        """英文分词，转小写+标点处理+语言标记后调用 SentencePiece。"""
        if self.sp is None:
            raise ValueError("BPE model not loaded")
        
        text = text.lower()
        text = re.sub(r'([.,!?;:])', r' \1', text)
        text = "▁en " + text
        pieces = self.sp.encode(text, out_type=str)
        
        result = [p for p in pieces if p not in ['▁zh', '▁en']]
        return result
    
    def encode(self, text, lang="zh", add_bos=True, add_eos=True):
        """
        文本转 ID 序列。
        lang: "zh" 或 "en"，返回如 [2, 1234, 5678, 3] (2=<s>, 3=</s>)。
        """
        if lang == "zh":
            tokens = self.tokenize_zh(text)
        else:
            tokens = self.tokenize_en(text)
        
        ids = []
        if add_bos:
            ids.append(self.bos_id)
        
        for token in tokens:
            piece_id = self.sp.piece_to_id(token)
            if piece_id == self.unk_id:
                ids.append(self.unk_id)
            else:
                ids.append(piece_id)
        
        if add_eos:
            ids.append(self.eos_id)
        
        return ids
    
    def decode(self, ids, lang="en"):
        """ID 序列还原为文本，过滤特殊 token，中文输出去空格。"""
        pieces = [self.sp.id_to_piece(i) for i in ids]

        # 过滤特殊 token 和语言标记
        filtered = [p for p in pieces if p not in [self.pad_token, self.bos_token, self.eos_token, self.unk_token, '▁zh', '▁en']]

        # 将SentencePiece的单词边界前缀'▁'恢复为空格，再做简单清理
        text = ''.join(filtered).replace('▁', ' ').strip()

        # 对中文语料，去掉多余空格（保持常用中文输出不带空格）
        # 简单策略：如果文本包含汉字，则移除空格
        if any('\u4e00' <= ch <= '\u9fff' for ch in text):
            text = text.replace(' ', '')

        return text
    
    def get_vocab_size(self):
        return self.sp.get_piece_size() if self.sp else 0
    
    def __len__(self):
        return self.get_vocab_size()


def build_tokenizer(zh_file, en_file, vocab_size=32000, model_prefix="./bpe_unified"):
    """构建 BPE 分词器。首次运行自动训练（约 2-3 分钟），后续复用。"""
    tokenizer = UnifiedBPETokenizer(model_prefix)
    tokenizer.train(zh_file, en_file, vocab_size, model_prefix)
    return tokenizer


class BilingualTokenizer(UnifiedBPETokenizer):
    """兼容旧接口的别名"""
    pass
