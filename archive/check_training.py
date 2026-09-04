"""训练状态检查"""
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tensorboard.backend.event_processing import event_accumulator
import os

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
eval_events = []
lr_events = []

try:
    eval_events = ea.Scalars('Eval/Loss')
except KeyError:
    pass

try:
    lr_events = ea.Scalars('Train/Learning_Rate')
except KeyError:
    pass

print('=' * 60)
print('训练状态')
print('=' * 60)
print(f'Total Train Steps: {len(train_events)}')
print(f'Total Eval Epochs: {len(eval_events)}')

if len(eval_events) > 0:
    print(f'\nLatest Eval Loss: {eval_events[-1].value:.4f}')
    if len(eval_events) >= 3:
        recent_eval = [e.value for e in eval_events[-3:]]
        print(f'Recent 3 Eval Loss: {[f"{x:.4f}" for x in recent_eval]}')

if len(lr_events) > 0:
    print(f'\nCurrent LR: {lr_events[-1].value:.2e}')

if len(train_events) >= 5:
    recent = [e.value for e in train_events[-5:]]
    print(f'\nRecent Train Loss (last 5): {[f"{x:.4f}" for x in recent]}')

print('\n' + '=' * 60)
