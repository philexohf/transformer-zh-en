# Transformer 中英机器翻译

纯手写实现 Transformer 论文《Attention Is All You Need》，用于中英机器翻译。
在 12GB 显存单卡上训练 11 epoch（100 万句对），BLEU 达 36.87。

## 项目特性

- **纯手写 Transformer** — 多头注意力、位置编码、掩码机制全部手写，不依赖 `nn.Transformer`
- **统一 BPE 分词** — 中英文共享 32K 词表，SentencePiece 训练（100 万句对）
- **AMP 混合精度** — 自动混合精度训练
- **余弦退火 + Warmup** — 稳定收敛
- **DDP 多卡支持** — 多卡并行训练
- **完整流水线** — CSV 清洗 → 数据采样 → 分词器训练 → 模型训练 → BLEU 评估

## 环境要求

| 依赖 | 版本（已验证） | 说明 |
|------|---------------|------|
| Python | 3.12 | |
| PyTorch | 2.5.1+cu124 | AMP 混合精度训练 |
| CUDA | 12.4 | GPU 训练 |
| GPU | NVIDIA (12GB) | 实测约 28 it/s |

```bash
# 推荐使用 conda 环境
conda create -n dl2llm python=3.12
conda activate dl2llm
pip install -r requirements.txt
```

## 快速开始

### 1. 数据准备

数据从魔搭下载（6.3GB CSV）：

👉 [WMT-Chinese-to-English-Machine-Translation-Training-Corpus](https://www.modelscope.cn/datasets/iic/WMT-Chinese-to-English-Machine-Translation-Training-Corpus/files/)

下载后将 CSV 文件放入 `./data/WMT-CN-to-EN/`，然后分三步处理：

#### 1.1 清洗原始 CSV

```bash
python tools/process_wmt.py \
  --input data/WMT-CN-to-EN/wmt_zh_en_training_corpus.csv \
  --output_dir data/wmt_processed
```

输出：`data/wmt_processed/wmt_zh_en_training_corpus.zh` + `.en`（约 2473 万句对）

#### 1.2 采样训练集与验证集

从全量语料随机采样 100 万训练句对 + 10 万验证句对，输出 `data/wmt_processed/train.zh`/`.en` 与 `valid.zh`/`.en`：

```bash
python -c "from tools.preprocess_pipeline import step2_sample_data
step2_sample_data('data/wmt_processed/wmt_zh_en_training_corpus.zh',
                  'data/wmt_processed/wmt_zh_en_training_corpus.en')"
```

采样数量可通过 `train_num=`/`valid_num=` 参数调整。

#### 1.3 训练 BPE 分词器

在训练集上训练中英文统一 BPE 分词器，输出 `checkpoints/bpe_unified.model` + `.vocab`：

```bash
python -c "from tools.preprocess_pipeline import step3_train_tokenizer
step3_train_tokenizer('data/wmt_processed/train.zh',
                      'data/wmt_processed/train.en')"
```

词表大小默认 32K，可通过 `vocab_size=` 参数调整。

以上三步也可一键执行（已存在的产物自动跳过）：

```bash
python tools/preprocess_pipeline.py
```

### 2. 训练

模型自动加载 `checkpoints/bpe_unified.model`（如不存在则自动训练）。

```bash
# 推荐配置：4 层轻量模型，~3 小时获得可用翻译
python train/train_llm.py \
  --data_dir data/wmt_processed \
  --epochs 11 \
  --batch_size 32 \
  --lr_multiplier 0.5 \
  --checkpoint_dir checkpoints \
  --num_encoder_layers 4 \
  --num_decoder_layers 4 \
  --d_model 384 \
  --d_ff 1536
```

训练过程（12GB 单卡实测）：约 28 it/s、单 epoch 约 18 分钟，11 epoch 总耗时约 3 小时；
train_loss 2.64 → val_loss 2.70；最佳模型自动保存至 `checkpoints/best_model.pt`。

#### 图1 Train Loss

![](./images/Train_Loss.jpg)

#### 图2 Train Learning Rate

![](./images/Train_Learning_Rate.jpg)

#### 图3 验证集 Loss（逐 epoch）

![](./images/Eval_train_loss.jpg)

多卡训练：

```bash
torchrun --nproc_per_node=3 train/train_llm.py
```

TensorBoard 实时曲线：`tensorboard --logdir checkpoints/runs`（浏览器打开 http://localhost:6006）

### 3. 评估

```bash
# 全量评估
python eval/evaluate_bleu.py --checkpoint ./checkpoints/best_model.pt

# 少量样本快速评估
python eval/evaluate_bleu.py --checkpoint ./checkpoints/best_model.pt --max_samples 100
```

| BLEU 分数 | 质量说明 |
|-----------|----------|
| 0-10 | 很差，模型未学习 |
| 10-20 | 一般，基础翻译 |
| 20-30 | 可用 |
| 30-40 | 较好，教学级 |
| 40+ | 优秀，商用级 |

### 4. 推理

提供两套推理方案：

- **`inference/infer.py`（FP32）** — 完整精度，支持 Beam Search
- **`inference/infer_quantized.py`（FP16）** — 半精度 GPU 推理，速度更快，体积仅 102MB

```bash
# FP32 推理（支持 Beam Search）
python inference/infer.py --input "这是一个简单的翻译模型。"
python inference/infer.py --beam_size 5

# 交互式推理
python inference/infer.py

# FP16 推理（需先导出，见下一节）
python inference/infer_quantized.py --input "这是一个简单的翻译模型。"
python inference/infer_quantized.py
```

#### 图4 推理结果展示

![](./images/zh_en.jpg)

### 5. FP16 量化导出

训练完成后，将 FP32 模型导出为 FP16 半精度，体积缩小 6 倍，推理速度更快。

```bash
python inference/quantize.py
```

导出结果：
- 输入：`checkpoints/best_model.pt`（FP32, 613 MB）
- 输出：`checkpoints/model_fp16.pt`（FP16, 102 MB）
- 自动验证 FP16 与 FP32 输出一致性

FP16 模型的交互式/单句推理用法与 FP32 相同（见上节推理命令）。

## 当前最佳结果

| 指标 | 值 | 说明 |
|------|-----|------|
| val_loss | 2.70 | epoch 11 |
| zh→en BLEU | 36.87 | 1000 样本，greedy decode |
| 总训练时间 | ~3 小时 | 11 epoch，12GB 单卡 |
| 训练数据 | 100 万句对 | 从 2473 万句对中采样 |
| 参数量 | 53.5M | 4 层 Transformer |

> **早停策略**：train-val gap 在 epoch 11 反转（train < val），此时停止训练可避免过拟合。

## 超参数

### 模型结构

| 参数 | 值 | 说明 |
|------|-----|------|
| d_model | 384 | 隐藏层维度（论文原版 512，消费级 GPU 优化） |
| nhead | 8 | 多头注意力头数 |
| num_encoder_layers | 4 | 编码器层数（论文原版 6，小数据防过拟合） |
| num_decoder_layers | 4 | 解码器层数 |
| d_ff | 1536 | 前馈网络维度（4 × d_model） |
| dropout | 0.1 | Dropout 比率 |
| 参数量 | 53.5M | 论文原版 65M |

### 训练配置

| 参数 | 值 | 说明 |
|------|-----|------|
| batch_size | 32 | 单卡批次大小 |
| accumulate_grad | 1 | 梯度累积步数 |
| epochs | 11 | 目标训练轮数 |
| warmup_steps | 4000 | 学习率预热步数 |
| lr_multiplier | 0.5 | 学习率乘数（余弦退火） |
| label_smoothing | 0.1 | 标签平滑 |
| clip_grad | 1.0 | 梯度裁剪阈值 |

### 数据配置

| 参数 | 值 | 说明 |
|------|-----|------|
| vocab_size | 32,000 | 中英文统一 BPE 词表 |
| max_len | 128 | 最大序列长度 |

## 项目结构

```
Transformer_zh_en2026/
├── core/                        # 核心库（共享代码包：配置/分词/数据/模型）
│   ├── __init__.py              #   统一导出（含 build_model）
│   ├── config.py                # 超参定义与默认值
│   ├── tokenizer.py             # 统一 BPE 分词器
│   ├── dataset.py               # 数据集（TranslationDataset + collate_fn）
│   └── transformer.py           # 纯手写 Transformer 实现
│
├── train/                       # 训练轨
│   ├── train_llm.py             # 训练脚本（AMP + CosineLR + AdamW）【推荐使用】
│   └── train_2017.py            # 论文原版训练脚本（保留参考）
│
├── inference/                   # 推理与导出轨
│   ├── infer.py                 # FP32 推理（Greedy / Beam Search）
│   ├── infer_quantized.py       # FP16 半精度推理
│   └── quantize.py              # FP16 模型导出
│
├── eval/                        # 评测轨
│   └── evaluate_bleu.py         # BLEU 评估脚本
│
├── tools/                       # 项目工具（数据准备 / 分词器 / 生成演示 / 配置）
│   ├── preprocess_pipeline.py   # 数据预处理流水线
│   ├── process_wmt.py           # WMT 原始 CSV → 清洗文本
│   ├── process_subset.py        # 子集数据预处理（去中文空格等）
│   ├── tokenize_text.py         # 分词演示与交互工具
│   ├── train_tokenizer_run.py   # 分词器训练入口
│   ├── generate_samples.py      # 贪心解码批量生成
│   ├── generate_beam.py         # Beam Search 批量生成
│   ├── generate_sampling.py     # 采样批量生成（temperature / top-k）
│   ├── print_config.py          # 打印默认超参
│   └── README.md                # 工具使用说明
├── archive/                     # 历史训练诊断脚本（已退役，仅供回溯）
├── tests/                       # pytest 单元测试（运行: python -m pytest tests -v）
├── checkpoints/                 # 模型权重与 BPE 产物（权重不入仓库）
├── data/                        # 语料（wmt_processed 不入仓库；debug_small 调试数据）
├── images/                      # 训练曲线与推理效果图
├── README.md
└── requirements.txt
```

## 核心实现

### 1. 模型结构（core/transformer.py）

| 组件 | 实现要点 |
|------|---------|
| PositionalEncoding | 固定正弦/余弦位置编码，`PE(pos,2i)=sin(pos/10000^(2i/d))`，序列超长时动态扩展 |
| Scaled Dot-Product Attention | `softmax(QK^T / √d_k)V`，`-inf` 掩码，NaN 兜底 |
| MultiHeadAttention | 8 头并行，独立 Q/K/V/O 线性投影，`d_k = d_model / nhead` |
| PositionWiseFFN | `Linear → ReLU → Dropout → Linear` |
| AddNorm | Post-LN（残差连接后 LayerNorm，与原论文一致） |
| EncoderLayer | Self-Attn → AddNorm → FFN → AddNorm |
| DecoderLayer | Masked Self-Attn → Cross-Attn → FFN，三层 AddNorm |
| Transformer | 独立 src/tgt embed，Xavier 初始化，`encode/decode/forward` 三入口 |

### 2. 分词器（core/tokenizer.py）

- 中英文共享 32K BPE 词表（SentencePiece）
- 语言标记 `▁zh`/`▁en` 让 BPE 学习语言特定的子词分布
- 中文去空格直编，英文小写 + 标点分离后编码
- 解码自动判断中文去掉额外空格

### 3. 训练方案

| 方案 | 精度 | 优化器 | LR 调度 | 适用场景 |
|------|------|--------|---------|---------|
| `train/train_llm.py`（推荐） | AMP 混合精度 | AdamW（wd=0.01） | CosineAnnealing + Warmup | 消费级 GPU 优化 |
| `train/train_2017.py`（参考） | FP32 | Adam | `d^-0.5 · min(step^-0.5, step · warmup^-1.5)` | 论文复现 |

### 4. 推理与解码

| 策略 | 文件 | 算法要点 |
|------|------|---------|
| Greedy | `inference/infer.py` / `inference/infer_quantized.py` | 每步 `argmax`，到 `eos` 停止 |
| Beam Search | `inference/infer.py` / `tools/generate_beam.py` | 宽度 5 + length penalty α=0.6 + n-gram 去重 |
| Sampling | `tools/generate_sampling.py` | Temperature / top-k / n-gram 回退到 argmax |

### 5. 量化导出（inference/quantize.py）

- FP32（613 MB）→ FP16（102 MB），精度无损验证
- 自描述导出格式：`{'model_state_dict': ..., 'model_config': {...}}`
- 导出后可用 `inference/infer_quantized.py` 独立推理，无需 `best_model.pt`

### 6. 数据流水线

```
CSV(6.3GB) → tools/process_wmt.py → 2473万句对
  → sample → 100万 train + 10万 valid
    → tools/train_tokenizer_run.py → 32K BPE 词表
```

## 参考文献

- Attention Is All You Need (Vaswani et al., 2017) — https://arxiv.org/abs/1706.03762

---

## 许可证

本项目采用 MIT 许可证（详见项目根目录 LICENSE 文件）。
项目允许自由使用、修改、商用与分发，使用过程中请保留 LICENSE 文件及原始版权信息。
