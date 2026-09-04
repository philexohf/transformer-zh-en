"""Check loss details around the explosion point"""
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob
import os

event_files = sorted(glob.glob('checkpoints/runs/events.out.tfevents.*'))
print(f'Found {len(event_files)} event files')
for f in event_files:
    ea = EventAccumulator(f)
    ea.Reload()
    if 'Train/Loss' in ea.Tags()['scalars']:
        t = ea.Scalars('Train/Loss')
        print(f'  {os.path.basename(f)}: {len(t)} events, steps {t[0].step}-{t[-1].step}')

# Use the latest file
latest = event_files[-1]
ea = EventAccumulator(latest)
ea.Reload()
train = ea.Scalars('Train/Loss')

# Find gaps > 500 steps
print('\n--- Large gaps in step sequence ---')
prev_step = None
for e in train:
    if prev_step and e.step - prev_step > 200:
        print(f'  Gap: step {prev_step} -> {e.step} ({e.step - prev_step} steps missing)')
    prev_step = e.step

# Show last healthy and first unhealthy
print('\n--- Transition from healthy to unhealthy ---')
last_healthy = None
for e in train:
    if e.value < 5:
        last_healthy = (e.step, e.value)
if last_healthy:
    print(f'Last loss < 5: step {last_healthy[0]} = {last_healthy[1]:.4f}')
else:
    print('No loss < 5 found')

first_bad = None
for e in train:
    if e.value > 10:
        first_bad = (e.step, e.value)
        break
if first_bad:
    print(f'First loss > 10: step {first_bad[0]} = {first_bad[1]:.4f}')
else:
    print('No loss > 10 found')

# Show the gap zone
if last_healthy and first_bad:
    print(f'\n--- Steps around gap ({last_healthy[0]}-{first_bad[0]}) ---')
    for e in train:
        if last_healthy[0] <= e.step <= first_bad[0]:
            print(f'  Step {e.step}: loss={e.value:.4f}')
