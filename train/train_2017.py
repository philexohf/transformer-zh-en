"""
Transformer 训练脚本（论文原版实现）

注意：此脚本保留用于参考，推荐使用 train/train_llm.py 进行训练。
train/train_llm.py 提供了更好的性能：AMP混合精度 + CosineLR学习率调度 + AdamW优化器

用法: python train/train_2017.py
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
try:
    from torch.utils.tensorboard.writer import SummaryWriter
    TB_AVAILABLE = True
except ImportError:
    SummaryWriter = None
    TB_AVAILABLE = False
import os
import random
import numpy as np
from tqdm import tqdm

from core.config import get_args
from core.transformer import Transformer
from core.tokenizer import build_tokenizer
from core.dataset import TranslationDataset, collate_fn


def build_model(vocab_size, config):
    """构建Transformer模型"""
    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=config.d_model,
        num_heads=config.nhead,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        d_ffn=config.d_ff,
        dropout=config.dropout,
        max_len=config.max_len,
        pad_idx=0
    )
    return model


def set_seed(seed):
    """固定随机种子，确保实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_lr(step, d_model, warmup_steps):
    """
    论文原版学习率调度公式：
    lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
    """
    step = max(1, step)
    base_lr = d_model ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))
    return base_lr


def train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device, epoch, args, global_step, writer):
    """训练一个 epoch，包含前向、损失计算、反向传播和参数更新。"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    # 梯度累积：模拟更大 batch size
    accumulate_grad = getattr(args, 'accumulate_grad', 1)
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}") if args.local_rank == 0 else train_loader
    
    for batch_idx, (src, tgt) in enumerate(pbar):
        src = src.to(device)
        tgt = tgt.to(device)
        
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        logits = model(src, tgt_input)
        
        # 用 reshape 而非 view：Transformer 内 tensor 可能不连续，view 会报错
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
        
        loss = loss / accumulate_grad
        
        loss.backward()
        
        if (batch_idx + 1) % accumulate_grad == 0 or (batch_idx + 1) == len(train_loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            if args.local_rank == 0 and writer is not None:
                current_loss = loss.item() * accumulate_grad
                current_lr = scheduler.get_last_lr()[0]
                writer.add_scalar('Train/Loss', current_loss, global_step)
                writer.add_scalar('Train/Learning_Rate', current_lr, global_step)
            global_step += 1
        
        total_loss += loss.item() * accumulate_grad
        num_batches += 1
        
        if args.local_rank == 0 and batch_idx % args.log_interval == 0:
            lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({
                "loss": f"{loss.item() * accumulate_grad:.4f}", 
                "lr": f"{lr:.6f}"
            })
    
    return total_loss / num_batches, global_step


def evaluate(model, val_loader, criterion, device):
    """验证集评估，不计算梯度，不使用梯度累积。"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for src, tgt in val_loader:
            src = src.to(device)
            tgt = tgt.to(device)
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            logits = model(src, tgt_input)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def main():
    """训练入口：解析参数 -> 初始化 DDP -> 构建数据/模型 -> 训练循环 -> 保存。"""
    args = get_args()
    
    # DDP 初始化
    # 单卡: RANK=-1, WORLD_SIZE=1; 多卡: RANK=0/1/2, WORLD_SIZE=3
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.rank = int(os.environ['RANK'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
    else:
        args.world_size = 1
        args.rank = 0
        args.local_rank = 0
    
    if args.world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
    
    device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    
    if args.local_rank == 0:
        print(f"Training with {args.world_size} GPUs")
        print(f"Config: d_model={args.d_model}, nhead={args.nhead}, layers={args.num_encoder_layers}")
    
    # 分词器
    tokenizer = build_tokenizer(
        os.path.join(args.data_dir, "train.zh"),
        os.path.join(args.data_dir, "train.en"),
        args.vocab_size,
        os.path.join(args.checkpoint_dir, "bpe_unified")
    )

    # 数据集
    train_dataset = TranslationDataset(args.data_dir, tokenizer, args.max_len, "train")
    val_dataset = TranslationDataset(args.data_dir, tokenizer, args.max_len, "valid")
    
    if args.world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=args.rank)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            sampler=train_sampler, 
            collate_fn=collate_fn
        )
        val_sampler = DistributedSampler(val_dataset, num_replicas=args.world_size, rank=args.rank)
        val_loader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            sampler=val_sampler, 
            collate_fn=collate_fn
        )
    else:
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            collate_fn=collate_fn,
            num_workers=2,
            pin_memory=True
        )
    
    # 模型
    model = build_model(len(tokenizer), args).to(device)
    if args.world_size > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

    # 损失函数
    criterion = nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_id,
        label_smoothing=args.label_smoothing
    )

    # 优化器
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=args.lr, 
        betas=(0.9, 0.98), 
        eps=1e-9
    )

    # 学习率调度（论文原版 Warmup）
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: get_lr(step, args.d_model, args.warmup_steps)
    )

    # 断点续训
    start_epoch = 0
    if args.load_checkpoint and os.path.exists(args.load_checkpoint):
        if args.local_rank == 0:
            print(f"Loading checkpoint: {args.load_checkpoint}")
        checkpoint = torch.load(args.load_checkpoint, map_location=device, weights_only=False)
        if args.world_size > 1:
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    
    # 创建检查点目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # TensorBoard
    writer = None
    if args.local_rank == 0:
        if TB_AVAILABLE:
            log_dir = os.path.join(args.checkpoint_dir, "runs")
            os.makedirs(log_dir, exist_ok=True)
            writer = SummaryWriter(log_dir)
            print(f"TensorBoard logging to: {log_dir}")
        else:
            print("Warning: tensorboard not available, skipping logging")
    
    best_loss = float('inf')
    global_step = 0
    
    # 训练
    for epoch in range(start_epoch, args.epochs):
        # DDP: 每个epoch需要设置随机种子
        if args.world_size > 1:
            train_sampler.set_epoch(epoch)
        
        train_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, scheduler, 
            criterion, device, epoch, args, global_step, writer
        )
        
        if args.local_rank == 0:
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}")
            
            val_loss = evaluate(
                model.module if args.world_size > 1 else model, 
                val_loader, criterion, device
            )
            print(f"Epoch {epoch}: val_loss={val_loss:.4f}")
            
            if writer is not None:
                writer.add_scalar('Eval/Loss', val_loss, epoch)
                writer.add_scalar('Eval/train_loss', train_loss, epoch)
            
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if args.world_size > 1 else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'tokenizer': tokenizer,
                    'args': args
                }, os.path.join(args.checkpoint_dir, "best_model.pt"))
                print(f"Saved best model with val_loss={val_loss:.4f}")
            
            if (epoch + 1) % 5 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if args.world_size > 1 else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt"))
    
    if args.world_size > 1:
        dist.destroy_process_group()
    if writer is not None:
        writer.close()
    
    if args.local_rank == 0:
        print("Training completed!")
        print(f"View TensorBoard with: tensorboard --logdir {os.path.join(args.checkpoint_dir, 'runs')}")


if __name__ == "__main__":
    main()
