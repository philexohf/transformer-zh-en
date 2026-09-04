import os
from pathlib import Path

root = Path('.')
out = Path('data/debug_small/delete_candidates.txt')
out.parent.mkdir(parents=True, exist_ok=True)

candidates = []
for p in root.rglob('*'):
    if not p.is_file():
        continue
    fn = str(p).replace('\\', '/')
    name = p.name
    # checkpoint epoch files (exclude best_model.pt)
    if '/checkpoints/' in fn and name.startswith('checkpoint_epoch_') and fn.endswith('.pt'):
        candidates.append(p)
        continue
    if '/checkpoints/' in fn and '/runs/events.out.tfevents' in fn:
        candidates.append(p)
        continue
    # bpe duplicates (not the root checkpoints/bpe_unified.*)
    if name in ('bpe_unified.model', 'bpe_unified.vocab') and not fn.endswith('checkpoints/bpe_unified.model') and not fn.endswith('checkpoints/bpe_unified.vocab'):
        candidates.append(p)
        continue
    # debug outputs
    if fn.startswith('data/debug_small/') and (name.startswith('sample_predictions_') and name.endswith('.txt')):
        candidates.append(p)
        continue
    if fn.startswith('data/debug_small/') and (name.startswith('samples_') and name.endswith('.txt')):
        candidates.append(p)
        continue
    if fn.startswith('data/debug_small/') and (name.startswith('diag_') and name.endswith('.jsonl')):
        candidates.append(p)
        continue
    if fn.startswith('data/debug_small/') and name == 'training_topk_sampling_stats.jsonl':
        candidates.append(p)
        continue
    if fn.startswith('data/debug_small/') and name == 'exp_full_snapshot.txt':
        candidates.append(p)
        continue

candidates = sorted(set(candidates), key=lambda p: p.stat().st_size, reverse=True)
with open(out, 'w', encoding='utf-8') as f:
    for p in candidates:
        f.write(f"{p}\t{p.stat().st_size}\t{p.stat().st_mtime}\n")
    f.write('\n')
    f.write(f"TotalFiles: {len(candidates)}\n")
    total = sum(p.stat().st_size for p in candidates)
    f.write(f"TotalSizeBytes: {total}\n")

print('Wrote', out)
