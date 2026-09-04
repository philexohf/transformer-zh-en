import os
import sys
import json
import torch
from torch.utils.data import DataLoader

# add project root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from tokenizer import UnifiedBPETokenizer
from models.transformer import Transformer
from dataset import TranslationDataset, collate_fn
from train_2017 import get_lr


def load_checkpoint(path, device):
    ck = torch.load(path, map_location=device)
    return ck


def main():
    checkpoint_path = os.environ.get('DIAG_CHECKPOINT', 'checkpoints/best_model.pt')
    data_dir = os.environ.get('DIAG_DATA', 'data/debug_small')
    out_file = os.environ.get('DIAG_OUT', 'data/debug_small/training_early_stats.jsonl')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ck = load_checkpoint(checkpoint_path, device)
    args_ck = ck.get('args', None)

    # load tokenizer (must match the tokenizer used during training)
    tokenizer = UnifiedBPETokenizer(os.path.join('checkpoints', 'bpe_unified'))

    # build model
    d_model = getattr(args_ck, 'd_model', 512)
    nhead = getattr(args_ck, 'nhead', 8)
    num_enc = getattr(args_ck, 'num_encoder_layers', 6)
    num_dec = getattr(args_ck, 'num_decoder_layers', 6)
    d_ff = getattr(args_ck, 'd_ff', 2048)
    dropout = getattr(args_ck, 'dropout', 0.1)
    max_len = getattr(args_ck, 'max_len', 200)

    model = Transformer(
        src_vocab_size=len(tokenizer),
        tgt_vocab_size=len(tokenizer),
        d_model=d_model,
        num_heads=nhead,
        num_encoder_layers=num_enc,
        num_decoder_layers=num_dec,
        d_ffn=d_ff,
        dropout=dropout,
        max_len=max_len,
        pad_idx=tokenizer.pad_id
    ).to(device)

    # load state_dict
    model.load_state_dict(ck['model_state_dict'])
    model.train()

    # dataset
    ds = TranslationDataset(data_dir, tokenizer, getattr(args_ck, 'max_len', 128), split='train')
    loader = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # loss
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id, label_smoothing=getattr(args_ck, 'label_smoothing', 0.0))

    # take first batch
    for src, tgt in loader:
        src = src.to(device)
        tgt = tgt.to(device)
        break

    tgt_input = tgt[:, :-1]
    tgt_output = tgt[:, 1:]

    # forward
    logits = model(src, tgt_input)

    loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))

    # backward to get gradients
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()
    loss.backward()

    # grad norm
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    # topk logits for first sample last position
    probs = torch.softmax(logits[0, -1].detach().cpu(), dim=-1)
    topk = torch.topk(probs, k=10)
    topk_ids = topk.indices.tolist()
    topk_vals = topk.values.tolist()

    # compute initial lr via paper formula (step=1)
    warmup = getattr(args_ck, 'warmup_steps', 4000)
    lr0 = get_lr(1, d_model, warmup)

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    rec = {
        'checkpoint': checkpoint_path,
        'lr_step1': lr0,
        'loss': float(loss.item()),
        'grad_norm': float(total_norm),
        'topk_ids': topk_ids,
        'topk_vals': topk_vals
    }

    with open(out_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    print('Wrote diagnostics to', out_file)


if __name__ == '__main__':
    main()
