"""Central configuration and reproducibility utilities for Whisper Yoruba LoRA ASR."""

from dataclasses import dataclass, field
import os
import random
from typing import Tuple

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set deterministic seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


@dataclass
class ProjectConfig:
    """Hyperparameters, dataset splits, and model settings."""

    # Reproducibility
    seed: int = 42

    # Model & Task
    model_name: str = "openai/whisper-small"
    language: str = "yo"
    task: str = "transcribe"

    # Dataset (BibleTTS Yoruba)
    dataset_name: str = "Imakandi-Labs/bibletss-yoruba"
    openslr_url: str = "https://www.openslr.org/resources/129/yoruba.tgz"
    val_books: Tuple[str, ...] = ("ROM", "1CO")
    test_books: Tuple[str, ...] = ("JHN",)

    # Audio constraints
    orig_sampling_rate: int = 48000
    target_sampling_rate: int = 16000
    min_duration_s: float = 1.0
    max_duration_s: float = 30.0

    # LoRA Configuration (Section 3.2)
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = ("q_proj", "v_proj")

    # SpecAugment Configuration (Section 2.3.4)
    apply_spec_augment: bool = True
    freq_mask_param: int = 27
    time_mask_param: int = 100

    # Optimization & Training (Section 3.3 & 3.4)
    learning_rate: float = 1e-4
    warmup_steps: int = 500
    weight_decay: float = 0.01
    label_smoothing_factor: float = 0.1
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 6
    fp16: bool = True
    output_dir: str = "checkpoints/whisper-small-yoruba-lora"

    # Theological Named Entity Error Audit (T-NEER) curated terms
    theological_entities: Tuple[str, ...] = field(
        default_factory=lambda: (
            "Jésù",
            "Krístì",
            "Ọlọ́run",
            "Olúwa",
            "Jerúsálẹ́mù",
            "Bẹ́tílẹ́hẹ́mù",
            "Pọ́ọ̀lù",
            "Pétérù",
            "Jòhánù",
            "Màríà",
            "Ábúráhámù",
            "Mósè",
            "Dáfídì",
            "Ìsírẹ́lì",
            "Gálílì",
            "Jùdíà",
            "ìgbàlà",
            "òdodo",
            "ìyè",
            "ìrònúpìwàdà",
            "ẹ̀ṣẹ̀",
            "oore-ọ̀fẹ́",
            "Ẹ̀mí",
            "Mímọ́",
        )
    )

