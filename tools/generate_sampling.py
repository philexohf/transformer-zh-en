import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.tokenizer import UnifiedBPETokenizer
from core.transformer import Transformer
from core.dataset import TranslationDataset, collate_fn
from torch.utils.data import DataLoader


def no_repeat_ngram_blocking(prev_tokens, cand_token, n):
    if n <= 1:
        return False
    seq = prev_tokens + [cand_token]
    if len(seq) < n:
        return False
    last = tuple(seq[-n:])
    for i in range(len(seq) - n):
        if tuple(seq[i:i+n]) == last:
            return True
    return False


def sample_next_token(logits, temperature=1.0, top_k=0):
    # logits: 1D tensor on CPU
    if temperature <= 0:
        # argmax
        return int(torch.argmax(logits).item())

    scores = logits / temperature
    if top_k is not None and top_k > 0:
        topk_vals, topk_idx = torch.topk(scores, k=min(top_k, scores.size(0)))
        probs = torch.softmax(topk_vals, dim=-1)
        choice = torch.multinomial(probs, num_samples=1).item()
        return int(topk_idx[choice].item())
    else:
        probs = torch.softmax(scores, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/best_model.pt')
    parser.add_argument('--data_dir', default='data/debug_small')
    parser.add_argument('--out', default='data/debug_small/sample_predictions_sampling.txt')
    parser.add_argument('--num', type=int, default=50)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=0)
    parser.add_argument('--no_repeat_ngram_size', type=int, default=0)
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

            tgt_ids = [tokenizer.bos_id]
            for _ in range(100):
                tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long).to(device)
                with torch.no_grad():
                    dec = model.decode(tgt_tensor, encoder_output, src_mask)
                    out = model.linear(dec)
                    logits = out[0, -1].cpu()

                # sample
                next_token = sample_next_token(logits, temperature=args.temperature, top_k=args.top_k)

                # no-repeat-ngram
                if args.no_repeat_ngram_size > 0 and no_repeat_ngram_blocking(tgt_ids, next_token, args.no_repeat_ngram_size):
                    # try fallback to top_k candidates
                    fallback_found = False
                    if args.top_k > 0:
                        topk = torch.topk(logits, k=min(args.top_k, logits.size(0)))
                        for cand in topk.indices.tolist():
                            if not no_repeat_ngram_blocking(tgt_ids, int(cand), args.no_repeat_ngram_size):
                                next_token = int(cand)
                                fallback_found = True
                                break
                    if not fallback_found:
                        # fallback to raw argmax
                        next_token = int(torch.argmax(logits).item())

                if next_token == tokenizer.eos_id:
                    break
                tgt_ids.append(next_token)

            pred = tokenizer.decode(tgt_ids[1:], lang='en')

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
