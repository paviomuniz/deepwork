"""BibleTTS Yoruba dataset loading, canonical book splitting, preprocessing, and collator."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import unicodedata

from datasets import Audio, Dataset, DatasetDict, load_dataset
import numpy as np
import torch
from transformers import WhisperProcessor

from src.config import ProjectConfig
from src.model import apply_spec_augment

# Standard 27 New Testament USFM 3-letter book codes
NT_BOOKS = frozenset(
    {
        "MAT",
        "MRK",
        "LUK",
        "JHN",
        "ACT",
        "ROM",
        "1CO",
        "2CO",
        "GAL",
        "EPH",
        "PHP",
        "COL",
        "1TH",
        "2TH",
        "1TI",
        "2TI",
        "TIT",
        "PHM",
        "HEB",
        "JAS",
        "1PE",
        "2PE",
        "1JN",
        "2JN",
        "3JN",
        "JUD",
        "REV",
    }
)


def normalize_nfc(text: str) -> str:
    """Enforce Unicode NFC normalization while preserving all Yoruba tone marks."""
    return unicodedata.normalize("NFC", text).strip()


def load_and_split_bibletts(
    config: ProjectConfig,
    nt_only: bool = True,
) -> DatasetDict:
    """Load BibleTTS Yoruba and apply canonical book-level train/val/test splits.

    Prevents data leakage across chapters/verses by splitting strictly on book IDs:
    - Test: Gospel of John (JHN)
    - Validation: Romans (ROM) & 1 Corinthians (1CO)
    - Train: Remaining New Testament books (or all remaining books if nt_only=False)
    """
    raw = load_dataset(config.dataset_name, split="train")
    raw = raw.cast_column("audio", Audio(sampling_rate=config.target_sampling_rate))

    val_set = set(config.val_books)
    test_set = set(config.test_books)

    def _filter_nt(example: Dict[str, Any]) -> bool:
        return example["book"] in NT_BOOKS

    if nt_only:
        raw = raw.filter(_filter_nt)

    train_ds = raw.filter(
        lambda ex: ex["book"] not in val_set and ex["book"] not in test_set
    )
    val_ds = raw.filter(lambda ex: ex["book"] in val_set)
    test_ds = raw.filter(lambda ex: ex["book"] in test_set)

    return DatasetDict({"train": train_ds, "validation": val_ds, "test": test_ds})


def filter_and_prepare_split(
    ds: Dataset,
    processor: WhisperProcessor,
    config: ProjectConfig,
    num_proc: int = 2,
) -> Dataset:
    """Filter audio by duration [1.0s, 30.0s] and extract log-mel features & token labels."""

    def _valid_duration(example: Dict[str, Any]) -> bool:
        audio = example["audio"]
        sr = audio["sampling_rate"]
        duration = len(audio["array"]) / float(sr)
        return config.min_duration_s <= duration <= config.max_duration_s

    ds = ds.filter(_valid_duration)

    def _prepare(batch: Dict[str, Any]) -> Dict[str, Any]:
        audio = batch["audio"]
        array = np.asarray(audio["array"], dtype=np.float32)

        # Normalize peak amplitude to [-1.0, 1.0] if needed
        max_abs = np.max(np.abs(array))
        if max_abs > 1.0:
            array = array / max_abs

        features = processor.feature_extractor(
            array, sampling_rate=audio["sampling_rate"]
        ).input_features[0]

        # Tensor shape hygiene (Section 5.1)
        assert features.shape == (
            80,
            3000,
        ), f"Expected log-mel shape (80, 3000), got {features.shape}"

        text = normalize_nfc(batch["text"])
        labels = processor.tokenizer(text).input_ids

        return {"input_features": features, "labels": labels}

    return ds.map(
        _prepare,
        remove_columns=ds.column_names,
        num_proc=num_proc,
    )


def load_tiny_sanity_batch(
    config: ProjectConfig,
    processor: WhisperProcessor,
    num_samples: int = 8,
) -> List[Dict[str, Any]]:
    """Stream a tiny batch of Yoruba BibleTTS samples for the Phase 1 1-batch sanity check."""
    stream = load_dataset(config.dataset_name, split="train", streaming=True)
    stream = stream.cast_column(
        "audio", Audio(sampling_rate=config.target_sampling_rate)
    )

    samples: List[Dict[str, Any]] = []
    for example in stream:
        audio = example["audio"]
        array = np.asarray(audio["array"], dtype=np.float32)
        sr = audio["sampling_rate"]
        duration = len(array) / float(sr)
        if not (config.min_duration_s <= duration <= config.max_duration_s):
            continue

        max_abs = np.max(np.abs(array))
        if max_abs > 1.0:
            array = array / max_abs

        features = processor.feature_extractor(array, sampling_rate=sr).input_features[0]
        assert features.shape == (
            80,
            3000,
        ), f"Expected log-mel shape (80, 3000), got {features.shape}"

        text = normalize_nfc(example["text"])
        labels = processor.tokenizer(text).input_ids
        samples.append(
            {
                "input_features": features,
                "labels": labels,
                "text": text,
                "verse_id": example.get("verse_id", "unknown"),
            }
        )
        if len(samples) >= num_samples:
            break

    return samples


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Data collator that pads Whisper log-mel features and masks padded labels with -100."""

    processor: Any
    decoder_start_token_id: int
    apply_spec_aug: bool = False
    freq_mask_param: int = 27
    time_mask_param: int = 100

    def __call__(
        self, features: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [
            {"input_features": f["input_features"]} for f in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # Shape assertion: [batch_size, 80, 3000]
        assert batch["input_features"].ndim == 3 and batch["input_features"].shape[1] == 80, (
            f"Unexpected input_features shape: {tuple(batch['input_features'].shape)}"
        )

        if self.apply_spec_aug:
            batch["input_features"] = apply_spec_augment(
                batch["input_features"],
                freq_mask_param=self.freq_mask_param,
                time_mask_param=self.time_mask_param,
            )

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

