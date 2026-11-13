"""Training & 3-way ablation orchestration using Hugging Face Seq2SeqTrainer."""

import argparse
import os
from typing import Dict

import torch
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from src.config import ProjectConfig, set_seed
from src.dataset import (
    DataCollatorSpeechSeq2SeqWithPadding,
    filter_and_prepare_split,
    load_and_split_bibletts,
)
from src.evaluate import make_compute_metrics
from src.model import build_processor, build_whisper_lora_model


def run_training(
    config: ProjectConfig,
    use_spec_augment: bool = True,
    max_steps: int = -1,
    processor=None,
    train_dataset=None,
    eval_dataset=None,
    test_dataset=None,
) -> Dict[str, float]:
    """Train Whisper-small + LoRA on BibleTTS Yoruba and evaluate on held-out test set."""
    set_seed(config.seed)
    config.apply_spec_augment = use_spec_augment

    if processor is None:
        processor = build_processor(config)
    model = build_whisper_lora_model(config, use_lora=True)

    if train_dataset is None or eval_dataset is None or test_dataset is None:
        raw_splits = load_and_split_bibletts(config, nt_only=True)
        train_dataset = filter_and_prepare_split(raw_splits["train"], processor, config)
        eval_dataset = filter_and_prepare_split(
            raw_splits["validation"], processor, config
        )
        test_dataset = filter_and_prepare_split(raw_splits["test"], processor, config)

    train_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
        apply_spec_aug=use_spec_augment,
        freq_mask_param=config.freq_mask_param,
        time_mask_param=config.time_mask_param,
    )
    eval_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
        apply_spec_aug=False,
    )

    run_suffix = "lora_specaug" if use_spec_augment else "lora_raw"
    output_dir = os.path.join(config.output_dir, run_suffix)

    use_fp16 = config.fp16 and torch.cuda.is_available()

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        label_smoothing_factor=config.label_smoothing_factor,
        num_train_epochs=config.num_train_epochs,
        max_steps=max_steps,
        lr_scheduler_type="cosine",
        fp16=use_fp16,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=25,
        predict_with_generate=True,
        generation_max_length=225,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        remove_unused_columns=False,
        report_to=["none"],
        seed=config.seed,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=train_collator,
        compute_metrics=make_compute_metrics(processor.tokenizer),
        processing_class=processor.feature_extractor,
    )

    trainer.train()

    # Evaluate on held-out Gospel of John test set without SpecAugment
    trainer.data_collator = eval_collator
    test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
    print(f"Test metrics ({run_suffix}):", test_metrics)

    adapter_dir = os.path.join(output_dir, "best_adapter")
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    print(f"Saved LoRA adapter and processor to {adapter_dir}")
    return test_metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune Whisper-small on BibleTTS Yoruba with LoRA."
    )
    parser.add_argument(
        "--no-specaug",
        action="store_true",
        help="Disable SpecAugment (Configuration B: LoRA without SpecAugment).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Optional max training steps override.",
    )
    args = parser.parse_args()

    config = ProjectConfig()
    run_training(
        config=config,
        use_spec_augment=not args.no_specaug,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()

