import sys
from collections import Counter

path = sys.argv[1] if len(sys.argv) > 1 else 'data/debug_small/sample_predictions_temp0.8_topk50.txt'

with open(path, 'r', encoding='utf-8') as f:
    lines = f.read().strip().splitlines()
if not lines:
    print('empty')
    sys.exit(1)

header = lines[0]
data = [line.split('\t') for line in lines[1:]]
preds = [parts[2].strip() if len(parts) > 2 else '' for parts in data]

total = len(preds)
tokens_counts = [len(p.split()) for p in preds]
avg_len = sum(tokens_counts) / total if total else 0

contains_the_the = sum(1 for p in preds if 'the the' in p)

# detect consecutive token repeats >=3
def has_consecutive_repeats(p):
    toks = p.split()
    if not toks:
        return False
    cnt = 1
    last = toks[0]
    for t in toks[1:]:
        if t == last:
            cnt += 1
            if cnt >= 3:
                return True
        else:
            last = t
            cnt = 1
    return False

consec_ge3 = sum(1 for p in preds if has_consecutive_repeats(p))

uniq_preds = len(set(preds))

token_counter = Counter()
bigram_counter = Counter()
for p in preds:
    toks = p.split()
    token_counter.update(toks)
    bigram_counter.update([' '.join(toks[i:i+2]) for i in range(len(toks)-1)])

print('file:', path)
print('total samples:', total)
print('avg tokens per pred: {:.2f}'.format(avg_len))
print('unique predictions:', uniq_preds)
print("contains 'the the':", contains_the_the, f"({contains_the_the/total:.2%})")
print('consecutive token repeats >=3:', consec_ge3, f"({consec_ge3/total:.2%})")
print('\nTop 15 tokens:')
for tok, c in token_counter.most_common(15):
    print(f'{tok}: {c}')

print('\nTop 15 bigrams:')
for bg, c in bigram_counter.most_common(15):
    print(f'{bg}: {c}')
