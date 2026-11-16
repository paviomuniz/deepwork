"""Phase 4 quantitative evaluation, T-NEER audit, and qualitative verse comparison."""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from peft import PeftModel
import torch
from torch.utils.data import DataLoader
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from src.config import ProjectConfig, set_seed
from src.dataset import (
    DataCollatorSpeechSeq2SeqWithPadding,
    filter_and_prepare_split,
    load_and_split_bibletts,
)
from src.evaluate import (
    compute_asr_metrics,
    compute_theological_entity_audit,
    normalize_yoruba_text,
)
from src.model import build_processor


def filter_raw_split_by_duration(raw_ds, config: ProjectConfig):
    """Apply the same [min_duration_s, max_duration_s] filter used in preprocessing."""

    def _valid_duration(example: Dict[str, Any]) -> bool:
        audio = example["audio"]
        sr = audio["sampling_rate"]
        duration = len(audio["array"]) / float(sr)
        return config.min_duration_s <= duration <= config.max_duration_s

    return raw_ds.filter(_valid_duration)


def load_eval_model(
    config: ProjectConfig,
    adapter_dir: Optional[str] = None,
) -> torch.nn.Module:
    """Load Whisper-small for inference, optionally attaching a saved LoRA adapter."""
    model = WhisperForConditionalGeneration.from_pretrained(config.model_name)
    model.generation_config.language = config.language
    model.generation_config.task = config.task
    model.generation_config.forced_decoder_ids = None
    model.config.use_cache = True

    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, adapter_dir)

    return model


@torch.no_grad()
def generate_predictions(
    model: torch.nn.Module,
    test_dataset,
    processor: WhisperProcessor,
    batch_size: int = 8,
    max_new_tokens: int = 225,
    device: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
    """Run batched Whisper generation on test_dataset and decode predictions & references."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
        apply_spec_aug=False,
    )
    dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    predictions: List[str] = []
    references: List[str] = []
    amp_ctx = (
        torch.amp.autocast("cuda", dtype=torch.float16)
        if device.startswith("cuda")
        else nullcontext()
    )

    for batch in dataloader:
        input_features = batch["input_features"].to(device)
        label_ids = batch["labels"].clone()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        with amp_ctx:
            generated_ids = model.generate(
                input_features=input_features,
                max_new_tokens=max_new_tokens,
            )

        pred_texts = processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        ref_texts = processor.tokenizer.batch_decode(
            label_ids, skip_special_tokens=True
        )
        predictions.extend(pred_texts)
        references.extend(ref_texts)

    return predictions, references


def evaluate_config_predictions(
    predictions: Sequence[str],
    references: Sequence[str],
    config: ProjectConfig,
) -> Dict[str, Any]:
    """Compute Task 4.1 (diacritic vs. stripped) and Task 4.2 (T-NEER) metrics."""
    asr = compute_asr_metrics(predictions, references)
    t_neer = compute_theological_entity_audit(
        predictions,
        references,
        entities=config.theological_entities,
    )
    return {
        **asr,
        "wer_diacritic_gap": round(asr["wer"] - asr["wer_no_tone"], 4),
        "cer_diacritic_gap": round(asr["cer"] - asr["cer_no_tone"], 4),
        **t_neer,
    }


def select_qualitative_samples(
    raw_test_ds,
    predictions_by_config: Dict[str, Sequence[str]],
    references: Sequence[str],
    config: ProjectConfig,
    num_samples: int = 5,
) -> List[Dict[str, Any]]:
    """Select representative verses across test books, prioritizing theological entities."""
    norm_entities = [normalize_yoruba_text(e).lower() for e in config.theological_entities]
    candidates: List[Tuple[int, int, str]] = []

    for idx in range(len(references)):
        ref_norm = normalize_yoruba_text(references[idx]).lower()
        matched = [e for e in norm_entities if e in ref_norm]
        book = raw_test_ds[idx].get("book", "UNK")
        candidates.append((len(matched), idx, book))

    # Sort by most theological entities first, then pick diverse books
    candidates.sort(key=lambda item: (-item[0], item[1]))
    selected_indices: List[int] = []
    seen_books = set()

    for _, idx, book in candidates:
        if book not in seen_books:
            selected_indices.append(idx)
            seen_books.add(book)
        if len(selected_indices) >= num_samples:
            break

    if len(selected_indices) < num_samples:
        for _, idx, _ in candidates:
            if idx not in selected_indices:
                selected_indices.append(idx)
            if len(selected_indices) >= num_samples:
                break

    samples: List[Dict[str, Any]] = []
    for idx in selected_indices:
        row = raw_test_ds[idx]
        ref_norm = normalize_yoruba_text(references[idx]).lower()
        matched_entities = [
            ent
            for ent in config.theological_entities
            if normalize_yoruba_text(ent).lower() in ref_norm
        ]
        samples.append(
            {
                "index": idx,
                "book": row.get("book", "UNK"),
                "chapter": row.get("chapter", ""),
                "verse": row.get("verse", ""),
                "verse_id": row.get("verse_id", f"{row.get('book', 'UNK')}_{idx}"),
                "ground_truth": references[idx],
                "matched_entities": matched_entities,
                "predictions": {
                    cfg_name: preds[idx]
                    for cfg_name, preds in predictions_by_config.items()
                },
            }
        )
    return samples


def format_phase4_markdown(
    metrics_by_config: Dict[str, Dict[str, Any]],
    qualitative_samples: Sequence[Dict[str, Any]],
) -> str:
    """Format Task 4.1, 4.2, and 4.3 outputs as a readable Markdown report."""
    lines = [
        "# Phase 4 Evaluation Report: Yoruba BibleTTS ASR",
        "",
        "## Task 4.1: Diacritic-Preserving vs. Diacritic-Stripped Breakdown",
        "",
        "| Configuration | WER (%) | WER No-Tone (%) | Diacritic WER Gap | CER (%) | CER No-Tone (%) | Diacritic CER Gap |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    labels = {
        "config_a_baseline": "Config A: Zero-Shot Baseline",
        "config_b_lora_raw": "Config B: LoRA (Raw)",
        "config_c_lora_specaug": "Config C: LoRA + SpecAugment",
    }

    for key, label in labels.items():
        if key not in metrics_by_config:
            continue
        m = metrics_by_config[key]
        lines.append(
            f"| {label} | {m['wer']:.2f} | {m['wer_no_tone']:.2f} | "
            f"+{m['wer_diacritic_gap']:.2f} | {m['cer']:.2f} | "
            f"{m['cer_no_tone']:.2f} | +{m['cer_diacritic_gap']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Task 4.2: Theological Named Entity Error Audit (T-NEER)",
            "",
            "| Configuration | Precision (%) | Recall (%) | F1 (%) | True Positives | False Negatives |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )

    for key, label in labels.items():
        if key not in metrics_by_config:
            continue
        m = metrics_by_config[key]
        lines.append(
            f"| {label} | {m['t_neer_precision']:.2f} | {m['t_neer_recall']:.2f} | "
            f"{m['t_neer_f1']:.2f} | {m['t_neer_tp']} | {m['t_neer_fn']} |"
        )

    lines.extend(
        [
            "",
            "## Task 4.3: Qualitative Side-by-Side Verse Comparisons",
            "",
        ]
    )

    for sample in qualitative_samples:
        lines.append(f"### {sample['verse_id']} (Book: `{sample['book']}`)")
        if sample["matched_entities"]:
            lines.append(
                f"- **Theological Entities:** {', '.join(sample['matched_entities'])}"
            )
        lines.append(f"- **Ground Truth:** {sample['ground_truth']}")
        for key, label in labels.items():
            if key in sample["predictions"]:
                lines.append(f"- **{label}:** {sample['predictions'][key]}")
        lines.append("")

    return "\n".join(lines)


def run_phase4_evaluation(
    config: ProjectConfig,
    lora_raw_dir: str,
    lora_specaug_dir: str,
    output_dir: str = "outputs",
    batch_size: int = 8,
    num_qualitative: int = 5,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute full Phase 4 evaluation across Config A, B, and C."""
    set_seed(config.seed)
    processor = build_processor(config)
    raw_splits = load_and_split_bibletts(config, nt_only=True)

    raw_test_filtered = filter_raw_split_by_duration(raw_splits["test"], config)
    test_ds = filter_and_prepare_split(
        raw_splits["test"], processor, config, num_proc=None
    )

    configs_to_eval = [
        ("config_a_baseline", None),
        ("config_b_lora_raw", lora_raw_dir),
        ("config_c_lora_specaug", lora_specaug_dir),
    ]

    metrics_by_config: Dict[str, Dict[str, Any]] = {}
    predictions_by_config: Dict[str, List[str]] = {}
    references: List[str] = []

    for cfg_name, adapter_dir in configs_to_eval:
        print(f"[Phase 4] Evaluating {cfg_name} (adapter={adapter_dir})...")
        model = load_eval_model(config, adapter_dir=adapter_dir)
        preds, refs = generate_predictions(
            model,
            test_ds,
            processor,
            batch_size=batch_size,
            device=device,
        )
        if not references:
            references = refs
        predictions_by_config[cfg_name] = preds
        metrics_by_config[cfg_name] = evaluate_config_predictions(
            preds, refs, config
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    qualitative = select_qualitative_samples(
        raw_test_filtered,
        predictions_by_config,
        references,
        config,
        num_samples=num_qualitative,
    )
    report_md = format_phase4_markdown(metrics_by_config, qualitative)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    results = {
        "metrics": metrics_by_config,
        "qualitative_samples": qualitative,
        "predictions": predictions_by_config,
        "references": references,
    }
    (out_path / "phase4_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_path / "phase4_report.md").write_text(report_md, encoding="utf-8")

    print(report_md)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 4 ASR & T-NEER Evaluation.")
    parser.add_argument(
        "--lora-raw-dir",
        default="checkpoints/whisper-small-yoruba-lora/lora_raw/best_adapter",
        help="Path to Config B (LoRA raw) best_adapter directory.",
    )
    parser.add_argument(
        "--lora-specaug-dir",
        default="checkpoints/whisper-small-yoruba-lora/lora_specaug/best_adapter",
        help="Path to Config C (LoRA + SpecAugment) best_adapter directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory to write phase4_results.json and phase4_report.md.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    run_phase4_evaluation(
        config=ProjectConfig(),
        lora_raw_dir=args.lora_raw_dir,
        lora_specaug_dir=args.lora_specaug_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
