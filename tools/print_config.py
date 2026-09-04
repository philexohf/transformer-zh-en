"""打印当前超参数配置"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.config import get_args

# 模拟解析默认参数
sys.argv = ['train_llm.py']
args = get_args()

print('=' * 60)
print('Transformer 中英翻译 - 当前超参数配置')
print('=' * 60)

print('\n【训练参数】')
print(f'  epochs          : {args.epochs}')
print(f'  batch_size      : {args.batch_size}')
print(f'  accumulate_grad : {args.accumulate_grad} (有效batch={args.batch_size * args.accumulate_grad})')

print('\n【学习率】')
print(f'  lr              : {args.lr}')
print(f'  lr_multiplier   : {args.lr_multiplier} (新增: 续训时可调大加速收敛)')
print(f'  warmup_steps    : {args.warmup_steps}')

print('\n【模型结构】')
print(f'  d_model         : {args.d_model}')
print(f'  nhead           : {args.nhead}')
print(f'  encoder_layers  : {args.num_encoder_layers}')
print(f'  decoder_layers  : {args.num_decoder_layers}')
print(f'  d_ff            : {args.d_ff}')

print('\n【正则化】')
print(f'  dropout         : {args.dropout}')
print(f'  label_smoothing : {args.label_smoothing}')
print(f'  clip_grad       : {args.clip_grad}')

print('\n【数据】')
print(f'  max_len         : {args.max_len}')
print(f'  vocab_size      : {args.vocab_size}')

print('\n【路径】')
print(f'  data_dir        : {args.data_dir}')
print(f'  checkpoint_dir  : {args.checkpoint_dir}')
print(f'  load_checkpoint : {args.load_checkpoint}')

print('\n【其他】')
print(f'  seed            : {args.seed}')
print(f'  log_interval    : {args.log_interval}')
print(f'  eval_interval   : {args.eval_interval}')
print(f'  save_interval   : {args.save_interval}')

print('\n' + '=' * 60)
print('【续训推荐命令】')
print('=' * 60)
print('python train_llm.py \\')
print('  --data_dir data/wmt_processed \\')
print('  --checkpoint_dir checkpoints \\')
print('  --epochs 70 \\')
print('  --load_checkpoint checkpoints/best_model.pt \\')
print('  --lr_multiplier 2.0')
print('=' * 60)
