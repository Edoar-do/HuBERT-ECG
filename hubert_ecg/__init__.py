from __future__ import annotations

from .configuration import HuBERTECGConfig, HuBERTECGClassificationConfig
from .modeling import HuBERTECG
from .modeling_classification import HuBERTECGForClassification
from transformers import AutoConfig, AutoModel

AutoConfig.register("hubert_ecg", HuBERTECGConfig)
AutoConfig.register("hubert_ecg_classification", HuBERTECGClassificationConfig)
AutoModel.register(HuBERTECGConfig, HuBERTECG)
AutoModel.register(HuBERTECGClassificationConfig, HuBERTECGForClassification)

__all__ = [
    "HuBERTECGConfig",
    "HuBERTECGClassificationConfig",
    "HuBERTECG",
    "HuBERTECGForClassification",
]
