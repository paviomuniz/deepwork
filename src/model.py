"""Whisper base model loading, LoRA adapter injection, and SpecAugment utilities."""

from typing import Tuple

from peft import LoraConfig, get_peft_model
import torch
import torchaudio.transforms as T
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from src.config import ProjectConfig


def build_processor(config: ProjectConfig) -> WhisperProcessor:
    """Load the WhisperProcessor configured for Yoruba transcription."""
    processor = WhisperProcessor.from_pretrained(
        config.model_name,
        language=config.language,
        task=config.task,
    )
    return processor


def apply_spec_augment(
    input_features: torch.Tensor,
    freq_mask_param: int = 27,
    time_mask_param: int = 100,
) -> torch.Tensor:
    """Apply SpecAugment (frequency & time masking) on log-mel spectrograms.

    Expected input shape: [batch_size, n_mels=80, time_steps=3000].
    """
    assert (
        input_features.ndim == 3 and input_features.shape[1] == 80
    ), f"Expected [batch, 80, time_steps], got {tuple(input_features.shape)}"

    freq_mask = T.FrequencyMasking(freq_mask_param=freq_mask_param)
    time_mask = T.TimeMasking(time_mask_param=time_mask_param)

    augmented = freq_mask(input_features)
    augmented = time_mask(augmented)
    return augmented


def count_parameters(model: torch.nn.Module) -> Tuple[int, int, float]:
    """Return (trainable_params, total_params, trainable_percentage)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = (100.0 * trainable / total) if total > 0 else 0.0
    return trainable, total, pct


def build_whisper_lora_model(
    config: ProjectConfig,
    use_lora: bool = True,
) -> torch.nn.Module:
    """Load Whisper-small and optionally wrap attention projections with LoRA."""
    model = WhisperForConditionalGeneration.from_pretrained(config.model_name)

    model.generation_config.language = config.language
    model.generation_config.task = config.task
    model.generation_config.forced_decoder_ids = None
    model.config.use_cache = False

    if use_lora:
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=list(config.lora_target_modules),
            lora_dropout=config.lora_dropout,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        trainable, total, pct = count_parameters(model)
        print(
            f"LoRA enabled: {trainable:,} / {total:,} trainable parameters ({pct:.2f}%)"
        )
    else:
        trainable, total, pct = count_parameters(model)
        print(
            f"Base model loaded (no LoRA): {trainable:,} / {total:,} trainable ({pct:.2f}%)"
        )

    return model

