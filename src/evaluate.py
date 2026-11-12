"""Evaluation metrics: WER, CER, and Theological Named Entity Error Audit (T-NEER)."""

import re
from typing import Callable, Dict, Iterable, List, Sequence
import unicodedata

import jiwer

from src.config import ProjectConfig


def normalize_yoruba_text(text: str) -> str:
    """Normalize text to Unicode NFC while preserving all tonal diacritics and underdots."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_diacritics(text: str) -> str:
    """Strip combining marks (NFD) for diagnostic diacritic-free comparison."""
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", stripped)


def compute_asr_metrics(
    predictions: Sequence[str],
    references: Sequence[str],
) -> Dict[str, float]:
    """Compute WER and CER (both diacritic-preserving and diacritic-stripped)."""
    norm_preds = [normalize_yoruba_text(p) for p in predictions]
    norm_refs = [normalize_yoruba_text(r) for r in references]

    # Filter out empty reference strings to avoid jiwer division by zero
    paired = [(p, r) for p, r in zip(norm_preds, norm_refs) if r]
    if not paired:
        return {"wer": 100.0, "cer": 100.0, "wer_no_tone": 100.0, "cer_no_tone": 100.0}

    p_list, r_list = zip(*paired)
    wer_score = 100.0 * jiwer.wer(list(r_list), list(p_list))
    cer_score = 100.0 * jiwer.cer(list(r_list), list(p_list))

    p_notone = [strip_diacritics(p) for p in p_list]
    r_notone = [strip_diacritics(r) for r in r_list]
    wer_notone = 100.0 * jiwer.wer(r_notone, p_notone)
    cer_notone = 100.0 * jiwer.cer(r_notone, p_notone)

    return {
        "wer": round(wer_score, 4),
        "cer": round(cer_score, 4),
        "wer_no_tone": round(wer_notone, 4),
        "cer_no_tone": round(cer_notone, 4),
    }


def compute_theological_entity_audit(
    predictions: Sequence[str],
    references: Sequence[str],
    entities: Iterable[str] | None = None,
) -> Dict[str, float]:
    """Compute precision, recall, and F1 on curated biblical proper nouns & theological terms."""
    if entities is None:
        entities = ProjectConfig().theological_entities

    norm_entities = [normalize_yoruba_text(e).lower() for e in entities]

    tp = 0
    fp = 0
    fn = 0

    for pred, ref in zip(predictions, references):
        pred_l = normalize_yoruba_text(pred).lower()
        ref_l = normalize_yoruba_text(ref).lower()

        for ent in norm_entities:
            in_ref = ent in ref_l
            in_pred = ent in pred_l
            if in_ref and in_pred:
                tp += 1
            elif in_pred and not in_ref:
                fp += 1
            elif in_ref and not in_pred:
                fn += 1

    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "t_neer_precision": round(100.0 * precision, 2),
        "t_neer_recall": round(100.0 * recall, 2),
        "t_neer_f1": round(100.0 * f1, 2),
        "t_neer_tp": tp,
        "t_neer_fn": fn,
    }


def make_compute_metrics(tokenizer) -> Callable:
    """Create a Hugging Face Seq2SeqTrainer-compatible compute_metrics function."""

    def compute_metrics(eval_pred) -> Dict[str, float]:
        pred_ids = eval_pred.predictions
        label_ids = eval_pred.label_ids

        # Replace -100 with pad_token_id before decoding
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        pred_str: List[str] = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str: List[str] = tokenizer.batch_decode(
            label_ids, skip_special_tokens=True
        )

        metrics = compute_asr_metrics(pred_str, label_str)
        return {"wer": metrics["wer"], "cer": metrics["cer"]}

    return compute_metrics

