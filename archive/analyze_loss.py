"""Loss变化趋势分析"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tensorboard.backend.event_processing import event_accumulator

events_dir = 'checkpoints/runs'
if not os.path.isdir(events_dir):
    print(f"Events directory not found: {events_dir}")
    sys.exit(1)
event_files = [f for f in os.listdir(events_dir) if f.startswith('events.out.tfevents')]
if not event_files:
    print(f"No event files found in {events_dir}")
    sys.exit(1)
event_files.sort(key=lambda x: os.path.getmtime(os.path.join(events_dir, x)), reverse=True)
latest = os.path.join(events_dir, event_files[0])

ea = event_accumulator.EventAccumulator(latest)
ea.Reload()

train_events = ea.Scalars('Train/Loss')
lr_events = []

try:
    lr_events = ea.Scalars('Train/Learning_Rate')
except KeyError:
    pass

print('=' * 80)
print('Loss 详细分析')
print('=' * 80)

if len(train_events) == 0:
    print('暂无训练数据')
    exit()

# 分段统计
total_steps = len(train_events)
segments = [
    (0, min(100, total_steps), "初始阶段 (0-100)"),
    (100, min(500, total_steps), "早期 (100-500)"),
    (500, min(1000, total_steps), "中期 (500-1000)"),
    (max(0, total_steps-200), total_steps, "最近 200 步"),
]

for start, end, name in segments:
    if start >= end or start >= total_steps:
        continue
    
    segment = train_events[start:end]
    losses = [e.value for e in segment]
    
    print(f'\n{name}:')
    print(f'  步数范围: {segment[0].step} - {segment[-1].step}')
    print(f'  平均 Loss: {sum(losses)/len(losses):.4f}')
    print(f'  最小 Loss: {min(losses):.4f}')
    print(f'  最大 Loss: {max(losses):.4f}')
    print(f'  标准差: {(sum((x - sum(losses)/len(losses))**2 for x in losses) / len(losses))**0.5:.4f}')

# LR 分析
if len(lr_events) > 0:
    print(f'\n学习率分析:')
    print(f'  当前 LR: {lr_events[-1].value:.2e}')
    if len(lr_events) >= 2:
        print(f'  初始 LR: {lr_events[0].value:.2e}')
        print(f'  LR 增长倍数: {lr_events[-1].value / lr_events[0].value:.1f}x')

print('\n' + '=' * 80)
print('分析结果:')
print('=' * 80)

recent_200 = [e.value for e in train_events[max(0, total_steps-200):]]
avg_recent = sum(recent_200) / len(recent_200)

first_100 = [e.value for e in train_events[:min(100, total_steps)]]
avg_first = sum(first_100) / len(first_100)

print(f'最初 100 步平均: {avg_first:.4f}')
print(f'最近 200 步平均: {avg_recent:.4f}')
print(f'变化: {avg_recent - avg_first:+.4f}')

if avg_recent > avg_first:
    print('\nLoss增加，可能原因:')
    print('  1. lr_multiplier=2.0 导致学习率过高')
    print('  2. Warmup 阶段正常波动')
    print('  3. 模型需要适应新的优化器 (AdamW)')
else:
    print('\nLoss在下降或稳定')
