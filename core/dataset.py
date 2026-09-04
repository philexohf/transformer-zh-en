"""
中英机器翻译数据集模块

实现PyTorch Dataset和DataLoader，支持平行语料加载、
BPE分词和动态padding。
"""

import torch
from torch.utils.data import Dataset
import os


class TranslationDataset(Dataset):
    """
    平行语料翻译数据集
    
    加载中英句子对，使用BPE分词，转换为ID序列，
    处理变长padding。
    
    数据结构：
        data_dir/
            train.zh  # 中文源句
            train.en  # 英文目标句
            valid.zh  # 验证集（中文）
            valid.en  # 验证集（英文）
    """
    
    def __init__(self, data_dir, tokenizer, max_len=200, split="train"):
        """
        初始化数据集
        
        Args:
            data_dir: 数据目录路径
            tokenizer: BPE分词器实例
            max_len: 最大序列长度（截断）
            split: 数据划分（"train", "valid", "test"）
        """
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # 读取中英平行语料
        zh_file = os.path.join(data_dir, f"{split}.zh")
        en_file = os.path.join(data_dir, f"{split}.en")
        
        # 如果验证集不存在，从训练集划分
        if not os.path.exists(zh_file) and split == "valid":
            print("Validation set not found, splitting from training set...")
            train_zh_file = os.path.join(data_dir, "train.zh")
            train_en_file = os.path.join(data_dir, "train.en")
            with open(train_zh_file, 'r', encoding='utf-8') as f:
                all_zh = f.readlines()
            with open(train_en_file, 'r', encoding='utf-8') as f:
                all_en = f.readlines()
            
            # 划分90%训练，10%验证
            split_idx = int(len(all_zh) * 0.9)
            self.zh_lines = all_zh[split_idx:]
            self.en_lines = all_en[split_idx:]
        else:
            with open(zh_file, 'r', encoding='utf-8') as f:
                self.zh_lines = f.readlines()
            with open(en_file, 'r', encoding='utf-8') as f:
                self.en_lines = f.readlines()
        
        # 确保中英文数据对齐
        assert len(self.zh_lines) == len(self.en_lines), "Data mismatch"
        
        print(f"Loaded {len(self.zh_lines)} {split} examples")
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.zh_lines)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        Returns:
            dict: {'src': 中文ID序列, 'tgt': 英文ID序列（包含<s>和</s>）}
        
        示例：
            src: [1234, 5678, 9012]  # 中文：你好世界
            tgt: [2, 3456, 7890, 3]  # 英文：<s>hello world</s>
        """
        zh_text = self.zh_lines[idx].strip()
        en_text = self.en_lines[idx].strip()
        
        # 中文编码：不需要<s>，需要</s>
        zh_ids = self.tokenizer.encode(zh_text, lang="zh", add_bos=False, add_eos=True)
        
        # 英文编码：需要<s>和</s>
        en_ids = self.tokenizer.encode(en_text, lang="en", add_bos=True, add_eos=True)
        
        # 截断超长序列，确保末尾为 EOS
        if len(zh_ids) > self.max_len:
            zh_ids = zh_ids[:self.max_len]
            zh_ids[-1] = self.tokenizer.eos_id
        if len(en_ids) > self.max_len:
            en_ids = en_ids[:self.max_len]
            en_ids[-1] = self.tokenizer.eos_id
        
        return {
            'src': torch.tensor(zh_ids, dtype=torch.long),
            'tgt': torch.tensor(en_ids, dtype=torch.long)
        }


def collate_fn(batch):
    """
    批次整理函数
    
    将多个样本合并成一个批次，并进行padding对齐。
    不同长度的句子用0填充到最长句子的长度。
    
    Args:
        batch: 多个样本的列表
    
    Returns:
        (src_batch, tgt_batch): 形状为 [batch_size, max_len] 的tensor
    
    示例：
        输入 (batch=2):
            src1: [1, 2, 3]         # 长度3
            src2: [1, 2, 3, 4, 5]   # 长度5
        输出:
            src: [[1,2,3,0,0],      # padding到长度5
                  [1,2,3,4,5]]
    """
    # 获取批次中每个样本的长度
    src_lens = [len(b['src']) for b in batch]
    tgt_lens = [len(b['tgt']) for b in batch]
    
    # 计算最大长度
    max_src_len = max(src_lens)
    max_tgt_len = max(tgt_lens)
    
    src_padded = []
    tgt_padded = []
    
    # 对每个样本进行padding
    for b in batch:
        # 源语言：末尾padding
        src = torch.cat([
            b['src'], 
            torch.zeros(max_src_len - len(b['src']), dtype=torch.long)
        ])
        # 目标语言：末尾padding
        tgt = torch.cat([
            b['tgt'], 
            torch.zeros(max_tgt_len - len(b['tgt']), dtype=torch.long)
        ])
        src_padded.append(src)
        tgt_padded.append(tgt)
    
    return torch.stack(src_padded), torch.stack(tgt_padded)