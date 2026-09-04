import os
import argparse
import torch
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.tokenizer import UnifiedBPETokenizer
from core.transformer import Transformer
from core.dataset import TranslationDataset, collate_fn
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/best_model.pt')
    parser.add_argument('--data_dir', default='data/debug_small')
    parser.add_argument('--out', default='data/debug_small/sample_predictions.txt')
    parser.add_argument('--num', type=int, default=50)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    if os.path.exists('checkpoints/bpe_unified.model'):
        tokenizer = UnifiedBPETokenizer('checkpoints/bpe_unified')
    else:
        tokenizer = UnifiedBPETokenizer()

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
                    decoder_output = model.decode(tgt_tensor, encoder_output, src_mask)
                    output = model.linear(decoder_output)
                    next_token = output[0, -1].argmax().item()
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
