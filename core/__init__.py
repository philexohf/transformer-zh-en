"""
核心库（core）— 配置 / 分词 / 数据 / 模型

参考 GleamLM 的核心包组织：共享代码按模块归入 core/ 包，
训练、推理、评测等入口脚本从各自目录导入本包。

用法:
    from core.config import get_args
    from core.tokenizer import UnifiedBPETokenizer
    from core.dataset import TranslationDataset, collate_fn
    from core.models.transformer import Transformer
"""
import sys

# pickle 兼容：历史 checkpoint 中的 tokenizer 对象按顶层模块名 "tokenizer"
# 序列化（UnifiedBPETokenizer 实例内嵌 SentencePiece proto）。
# 注册模块别名后，torch.load 旧权重时反序列化可自动解析到 core.tokenizer。
from . import tokenizer as _tokenizer_module

sys.modules.setdefault("tokenizer", _tokenizer_module)

from .config import Config, get_args
from .dataset import TranslationDataset, collate_fn
from .models import Transformer, build_model
from .tokenizer import BilingualTokenizer, UnifiedBPETokenizer, build_tokenizer

__all__ = [
    "Config",
    "get_args",
    "UnifiedBPETokenizer",
    "BilingualTokenizer",
    "build_tokenizer",
    "TranslationDataset",
    "collate_fn",
    "Transformer",
    "build_model",
]
