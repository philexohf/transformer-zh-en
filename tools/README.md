# Tools Directory

Project utility scripts: data preparation, tokenizer training, batch generation demos, and configuration display.
Retired diagnostic scripts from the original training phase live in `archive/` (see root `archive/`).

## Data Preparation

### process_wmt.py
Cleans raw WMT CSV into parallel corpus files (`wmt_zh_en_training_corpus.zh/.en`).

```bash
python tools/process_wmt.py --input data/WMT-CN-to-EN/wmt_zh_en_training_corpus.csv --output_dir data/wmt_processed
```

### preprocess_pipeline.py
One-key three-step pipeline: clean CSV → sample train/valid → train BPE tokenizer.

```bash
python tools/preprocess_pipeline.py
```

### process_subset.py
Subset data preprocessing (removes Chinese spaces, etc.).

### tokenize_text.py
Interactive tokenizer demo - shows pieces/ids for arbitrary Chinese or English text.

## Tokenizer Training

### train_tokenizer_run.py
Tokenizer retraining CLI entry with customizable parameters (vocab / output prefix).

```bash
python tools/train_tokenizer_run.py --zh data/wmt_processed/train.zh --en data/wmt_processed/train.en
```

> Note: `UnifiedBPETokenizer.train` loads an existing `.model` instead of retraining.
> Delete the target files first to force a retrain.

## Generation / Inference Demos

Batch decode over a dataset (default `data/debug_small`), writing translations to a txt file for quality review.

### generate_samples.py
Greedy decoding - picks the highest-probability token at each step. Fast but low diversity.

```bash
python tools/generate_samples.py --checkpoint checkpoints/best_model.pt
```

### generate_beam.py
Beam Search - keeps multiple candidate paths, producing higher-quality translations.

```bash
python tools/generate_beam.py --checkpoint checkpoints/best_model.pt --beam 5
```

### generate_sampling.py
Sampling-based generation (Temperature + Top-K). Controls randomness; useful for diversity analysis.

```bash
python tools/generate_sampling.py --checkpoint checkpoints/best_model.pt --temperature 0.8 --top_k 50
```

> For interactive single-sentence translation use `inference/infer.py` instead.

## Configuration

### print_config.py
Prints all current hyperparameters (epochs, batch_size, d_model, etc.).

```bash
python tools/print_config.py
```

## File Overview

| File | Size | Purpose |
|------|------|---------|
| preprocess_pipeline.py | 6.3 KB | Data preprocessing pipeline |
| process_wmt.py | 4.6 KB | WMT raw CSV → clean text |
| process_subset.py | 3.7 KB | Subset preprocessing |
| tokenize_text.py | 3.1 KB | Tokenizer demo / interactive tool |
| train_tokenizer_run.py | 0.9 KB | Tokenizer training CLI |
| generate_samples.py | 3.4 KB | Greedy batch decoding |
| generate_beam.py | 5.8 KB | Beam Search batch decoding |
| generate_sampling.py | 5.4 KB | Sampling batch decoding |
| print_config.py | 2.0 KB | Hyperparameter display |
