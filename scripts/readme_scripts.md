# Scripts Directory

Auxiliary tool scripts for generation/inference, tokenizer training, and configuration.
Historical training diagnostic scripts have been moved to `archive/` (see bottom).

## Generation / Inference

### generate_samples.py
Greedy decoding - picks the highest-probability token at each step. Fast but low diversity.

```bash
python scripts/generate_samples.py --checkpoint checkpoints/best_model.pt
```

### generate_beam.py
Beam Search - keeps multiple candidate paths, producing higher-quality translations.

```bash
python scripts/generate_beam.py --checkpoint checkpoints/best_model.pt --beam 5
```

### generate_sampling.py
Sampling-based generation (Temperature + Top-K). Controls randomness; useful for diversity analysis.

```bash
python scripts/generate_sampling.py --checkpoint checkpoints/best_model.pt --temperature 0.8 --top_k 50
```

---

## Tokenizer Training

### train_tokenizer_run.py
SentencePiece BPE tokenizer training entry point with customizable parameters.

```bash
python scripts/train_tokenizer_run.py --zh data/wmt_processed/train.zh --en data/wmt_processed/train.en
```

---

## Configuration

### print_config.py
Prints all current hyperparameters (epochs, batch_size, d_model, etc.).

```bash
python scripts/print_config.py
```

---

## Archive

`archive/` contains retired one-off diagnostic scripts from the original training phase
(check_training / analyze_loss / check_loss_detail / diagnose_dynamics /
analyze_predictions / generate_delete_candidates). They are kept for history only;
their assertions and checks have been superseded by `tests/`.

## File Overview

| File | Size | Purpose |
|------|------|---------|
| generate_samples.py | 3.4 KB | Greedy decoding |
| generate_beam.py | 5.8 KB | Beam Search generation |
| generate_sampling.py | 5.4 KB | Sampling-based generation |
| print_config.py | 2.0 KB | Hyperparameter display |
| train_tokenizer_run.py | 0.9 KB | Tokenizer training |
