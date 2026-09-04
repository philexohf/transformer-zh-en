import os
import sys
import argparse
import torch
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.tokenizer import UnifiedBPETokenizer
from core.models.transformer import Transformer
from core.dataset import TranslationDataset, collate_fn
from torch.utils.data import DataLoader


def no_repeat_ngram_blocking(prev_tokens, cand_token, n):
    if n <= 1:
        return False
    seq = prev_tokens + [cand_token]
    if len(seq) < n:
        return False
    # check if last n gram already appeared
    last = tuple(seq[-n:])
    for i in range(len(seq) - n):
        if tuple(seq[i:i+n]) == last:
            return True
    return False


def beam_search(model, encoder_output, src_mask, tokenizer, device, beam_size=5, max_len=100,
                no_repeat_ngram_size=3, length_penalty=0.6):
    vocab_size = tokenizer.get_vocab_size()
    bos = tokenizer.bos_id
    eos = tokenizer.eos_id

    beams = [([bos], 0.0, False)]  # (tokens, score, finished)

    for step in range(max_len):
        all_candidates = []
        for tokens, score, finished in beams:
            if finished:
                all_candidates.append((tokens, score, True))
                continue

            tgt = torch.tensor([tokens], dtype=torch.long).to(device)
            with torch.no_grad():
                dec = model.decode(tgt, encoder_output, src_mask)
                out = model.linear(dec)  # [1, seq_len, vocab]
                logp = torch.log_softmax(out[0, -1], dim=-1).cpu()

            # pick topk candidates for this beam
            topk = torch.topk(logp, k=beam_size)
            for idx, lp in zip(topk.indices.tolist(), topk.values.tolist()):
                # no-repeat-ngram
                if no_repeat_ngram_size > 0 and no_repeat_ngram_blocking(tokens, idx, no_repeat_ngram_size):
                    continue
                new_tokens = tokens + [idx]
                new_score = score + lp
                new_finished = (idx == eos)
                all_candidates.append((new_tokens, new_score, new_finished))

        # select top beam_size
        all_candidates.sort(key=lambda x: x[1] / ((max(len(x[0]) - 1, 1) ** length_penalty)), reverse=True)
        beams = all_candidates[:beam_size]

        # stop if all finished
        if all(f for _, _, f in beams):
            break

    # choose best finished beam; if none finished, choose best by score
    finished_beams = [b for b in beams if b[2]]
    if finished_beams:
        best = max(finished_beams, key=lambda x: x[1] / (max(len(x[0]) - 1, 1) ** length_penalty))
    else:
        best = max(beams, key=lambda x: x[1] / (max(len(x[0]) - 1, 1) ** length_penalty))

    return best[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/best_model.pt')
    parser.add_argument('--data_dir', default='data/debug_small')
    parser.add_argument('--out', default='data/debug_small/sample_predictions_beam.txt')
    parser.add_argument('--num', type=int, default=50)
    parser.add_argument('--beam', type=int, default=5)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # load tokenizer (BPE路径从checkpoint目录推导)
    bpe_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    bpe_prefix = os.path.join(bpe_dir, 'bpe_unified')
    if os.path.exists(bpe_prefix + '.model'):
        tokenizer = UnifiedBPETokenizer(bpe_prefix)
    else:
        tokenizer = UnifiedBPETokenizer()

    # load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_args = checkpoint.get('args', None)
    if model_args is None:
        from core.config import Config
        model_args = Config()

    vocab_size = tokenizer.get_vocab_size()
    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=model_args.d_model,
        num_heads=model_args.nhead,
        num_encoder_layers=model_args.num_encoder_layers,
        num_decoder_layers=model_args.num_decoder_layers,
        d_ffn=model_args.d_ff,
        dropout=model_args.dropout,
        max_len=model_args.max_len,
        pad_idx=0
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # dataset
    val_dataset = TranslationDataset(args.data_dir, tokenizer, model_args.max_len, 'valid')
    loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as fw:
        fw.write('orig_src\torig_tgt\tpred\n')
        count = 0
        for src, tgt in loader:
            src = src.to(device)
            src_ids = src[0].tolist()
            src_ids = [x for x in src_ids if x != tokenizer.pad_id]

            with torch.no_grad():
                encoder_output, src_mask = model.encode(torch.tensor([src_ids], dtype=torch.long).to(device))

            beam_ids = beam_search(model, encoder_output, src_mask, tokenizer, device,
                                   beam_size=args.beam, max_len=100, no_repeat_ngram_size=3)

            # remove bos
            if beam_ids and beam_ids[0] == tokenizer.bos_id:
                beam_out = beam_ids[1:]
            else:
                beam_out = beam_ids

            pred = tokenizer.decode(beam_out, lang='en')

            tgt_ids_ref = tgt[0].tolist()
            tgt_ids_ref = [x for x in tgt_ids_ref if x not in [tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id]]
            ref = tokenizer.decode(tgt_ids_ref, lang='en')
            src_text = tokenizer.decode([x for x in src_ids if x not in [tokenizer.pad_id]], lang='zh')

            fw.write(f"{src_text}\t{ref}\t{pred}\n")
            count += 1
            if count >= args.num:
                break

    print('Wrote', args.out)


if __name__ == '__main__':
    main()
