"""
Transformer 中英机器翻译 - 配置参数

严格对标 2017 年论文《Attention Is All You Need》

包含所有训练和推理所需的超参数配置。

核心超参数：
    - d_model: 512 (论文默认值)
    - nhead: 8 (论文默认值)  
    - num_encoder_layers: 6 (论文默认值)
    - num_decoder_layers: 6 (论文默认值)
    - d_ff: 2048 (论文默认值)
"""

import argparse
import os


def get_args():
    """
    获取命令行参数
    可以通过 python train/train_llm.py --d_model 256 等方式覆盖默认值
    """
    parser = argparse.ArgumentParser(description='Transformer机器翻译训练配置')
    
    # 随机种子
    # 固定种子确保实验可复现
    parser.add_argument("--seed", type=int, default=42, help='随机种子')
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=11, help='训练轮数')
    parser.add_argument("--batch_size", type=int, default=32, help='批次大小')
    parser.add_argument("--accumulate_grad", type=int, default=4, 
                        help='梯度累积步数（有效batch=batch_size*accumulate_grad，续训建议8+）')
    
    # 学习率
    # 配合 LambdaLR 使用，base_lr=1.0 时实际 LR 完全由 scheduler 公式决定
    parser.add_argument("--lr", type=float, default=1.0, help='学习率（LambdaLR 的 base_lr，默认 1.0 表示由 scheduler 全权控制）')
    parser.add_argument("--lr_multiplier", type=float, default=1.0,
                        help='学习率乘数（论文原版为1.0，续训建议0.5-1.0，收敛后降低防梯度爆炸）')
    
    # 模型结构 (论文标准配置)
    # 论文原版: d_model=512, heads=8, layers=6, d_ff=2048
    parser.add_argument("--d_model", type=int, default=384, 
                        help='模型隐藏层维度（论文: 512，小数据建议384）')
    parser.add_argument("--nhead", type=int, default=8, 
                        help='多头注意力头数（论文: 8）')
    parser.add_argument("--num_encoder_layers", type=int, default=4, 
                        help='编码器层数（论文: 6，小数据建议4）')
    parser.add_argument("--num_decoder_layers", type=int, default=4, 
                        help='解码器层数（论文: 6，小数据建议4）')
    parser.add_argument("--d_ff", type=int, default=1536, 
                        help='前馈网络隐藏层维度（论文: 2048，小数据建议1536）')
    
    # 正则化
    parser.add_argument("--dropout", type=float, default=0.1, 
                        help='Dropout比例，防止过拟合（默认0.1）')
    parser.add_argument("--label_smoothing", type=float, default=0.1, 
                        help='标签平滑系数，提升泛化能力')
    
    # 数据处理
    parser.add_argument("--max_len", type=int, default=128, 
                        help='最大序列长度，超过则截断')
    
    # 学习率调度 (论文原版)
    # 论文公式: lr = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
    parser.add_argument("--warmup_steps", type=int, default=4000, 
                        help='学习率warmup步数（论文: 4000）')
    parser.add_argument("--clip_grad", type=float, default=1.0, 
                        help='梯度裁剪阈值，防止梯度爆炸')
    
    # 词表
    parser.add_argument("--vocab_size", type=int, default=32000, 
                        help='词表大小，BPE分词器使用')
    
    # 日志与保存
    parser.add_argument("--log_interval", type=int, default=100, 
                        help='日志打印间隔（步）')
    parser.add_argument("--eval_interval", type=int, default=1000, 
                        help='验证间隔（步）')
    parser.add_argument("--save_interval", type=int, default=5000, 
                        help='模型保存间隔（步）')
    
    # 路径
    parser.add_argument("--data_dir", type=str, default="./data/wmt_processed", 
                        help='数据目录')
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", 
                        help='检查点保存目录')
    parser.add_argument("--load_checkpoint", type=str, default=None, 
                        help='加载检查点路径（断点续训）')
    
    return parser.parse_args()


class Config:
    """
    默认配置类 - 用于推理时加载模型参数
    
    此处配置与命令行参数保持一致
    推理时会自动从检查点中加载这些参数
    """
    seed = 42
    epochs = 11
    batch_size = 32
    accumulate_grad = 4
    lr = 1.0  # LambdaLR 的 base_lr，1.0 表示由 scheduler 全权控制
    lr_multiplier = 1.0
    d_model = 384
    nhead = 8
    num_encoder_layers = 4
    num_decoder_layers = 4
    d_ff = 1536
    dropout = 0.1
    max_len = 128
    warmup_steps = 4000
    clip_grad = 1.0
    label_smoothing = 0.1
    vocab_size = 32000
    log_interval = 100
    eval_interval = 1000
    save_interval = 5000
    data_dir = "./data/wmt_processed"
    checkpoint_dir = "./checkpoints"
    load_checkpoint = None
    
    device = "cuda"


# 显存适配参考：RTX 4070Ti(12GB) -> batch=32, max_len=128, accum=4
#               RTX 4090(24GB) -> batch=64, max_len=200, accum=2
#               3xRTX 3090     -> batch=32, max_len=200, accum=1
