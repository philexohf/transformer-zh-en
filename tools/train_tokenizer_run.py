import argparse
import os
import sys
# Ensure project root is on sys.path when running from tools/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.tokenizer import UnifiedBPETokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--zh', required=True)
parser.add_argument('--en', required=True)
parser.add_argument('--vocab', type=int, default=32000)
parser.add_argument('--prefix', default='checkpoints/bpe_unified_retrain')
args = parser.parse_args()

os.makedirs(os.path.dirname(args.prefix), exist_ok=True)

print('Starting tokenizer training')
print(' zh:', args.zh)
print(' en:', args.en)
print(' vocab:', args.vocab)
print(' prefix:', args.prefix)

tokenizer = UnifiedBPETokenizer()
tokenizer.train(args.zh, args.en, vocab_size=args.vocab, model_prefix=args.prefix)

print('Tokenizer training finished. Model at', args.prefix + '.model')
