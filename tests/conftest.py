"""
pytest 共享配置

提供：
- 项目根目录 sys.path 注入（保证 import config/models/tokenizer/dataset 可用）
- tokenizer fixture（依赖 checkpoints/bpe_unified.model，缺失时跳过）
- checkpoint_path fixture（依赖 checkpoints/best_model.pt，缺失时跳过）
"""

import os
import sys

import pytest

# 项目根目录（tests/ 的上一级），保证从任意目录运行 pytest 都可导入
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture(scope="session")
def tokenizer():
    """加载统一 BPE 分词器。模型文件缺失时跳过相关测试。"""
    from core.tokenizer import UnifiedBPETokenizer

    bpe_prefix = os.path.join(ROOT, "checkpoints", "bpe_unified")
    tok = UnifiedBPETokenizer(bpe_prefix)
    if tok.sp is None:
        pytest.skip(f"BPE 模型不存在: {bpe_prefix}.model（请先运行 python train_tokenizer.py）")

    # 与推理脚本一致：从 SP 模型同步特殊 token id（防御默认值漂移）
    tok.pad_id = tok.sp.pad_id()
    tok.unk_id = tok.sp.unk_id()
    tok.bos_id = tok.sp.bos_id()
    tok.eos_id = tok.sp.eos_id()
    return tok


@pytest.fixture(scope="session")
def checkpoint_path():
    """训练好的模型检查点路径。文件缺失时跳过相关集成测试。"""
    path = os.path.join(ROOT, "checkpoints", "best_model.pt")
    if not os.path.exists(path):
        pytest.skip(f"checkpoint 不存在: {path}（请先运行 python train_llm.py）")
    return path
