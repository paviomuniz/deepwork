"""Companion Gradio app for live Yoruba BibleTTS Whisper + LoRA inference (HF Spaces / local)."""

import os
from typing import Dict, Tuple

import gradio as gr
import librosa
from peft import PeftModel
import torch
from transformers import WhisperForConditionalGeneration

from src.config import ProjectConfig
from src.evaluate import normalize_yoruba_text
from src.model import build_processor

CONFIG = ProjectConfig()
PROCESSOR = build_processor(CONFIG)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ADAPTER_SOURCES: Dict[str, str | None] = {
    "Config A: Zero-Shot Baseline": None,
    "Config B: LoRA (Raw, r=32)": os.environ.get(
        "LORA_RAW_REPO",
        "checkpoints/lora_raw",
    ),
    "Config C: LoRA + SpecAugment": os.environ.get(
        "LORA_SPECAUG_REPO",
        "checkpoints/lora_specaug",
    ),
}

_MODEL_CACHE: Dict[str, torch.nn.Module] = {}


def get_model(choice: str) -> torch.nn.Module:
    """Load and cache the selected Whisper / LoRA model."""
    if choice in _MODEL_CACHE:
        return _MODEL_CACHE[choice]

    base = WhisperForConditionalGeneration.from_pretrained(CONFIG.model_name)
    base.generation_config.language = CONFIG.language
    base.generation_config.task = CONFIG.task
    base.generation_config.forced_decoder_ids = None
    base.config.use_cache = True

    adapter_src = ADAPTER_SOURCES[choice]
    if adapter_src is not None:
        model = PeftModel.from_pretrained(base, adapter_src)
    else:
        model = base

    model = model.to(DEVICE).eval()
    _MODEL_CACHE[choice] = model
    return model


@torch.no_grad()
def transcribe_audio(audio_path: str | None, model_choice: str) -> Tuple[str, str]:
    """Transcribe 16 kHz Yoruba audio and highlight detected theological entities."""
    if not audio_path:
        return "Please upload or record a Yoruba audio clip.", "None"

    waveform, _ = librosa.load(audio_path, sr=CONFIG.target_sampling_rate, mono=True)
    inputs = PROCESSOR.feature_extractor(
        waveform,
        sampling_rate=CONFIG.target_sampling_rate,
        return_tensors="pt",
    )
    input_features = inputs.input_features.to(DEVICE)

    model = get_model(model_choice)
    generated_ids = model.generate(input_features=input_features, max_new_tokens=225)
    raw_text = PROCESSOR.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    transcript = normalize_yoruba_text(raw_text)

    lowered = transcript.lower()
    found_entities = [
        ent
        for ent in CONFIG.theological_entities
        if normalize_yoruba_text(ent).lower() in lowered
    ]
    entity_summary = ", ".join(found_entities) if found_entities else "None detected"
    return transcript, entity_summary


demo = gr.Interface(
    fn=transcribe_audio,
    inputs=[
        gr.Audio(sources=["microphone", "upload"], type="filepath", label="Yoruba Scripture Audio"),
        gr.Radio(
            choices=list(ADAPTER_SOURCES.keys()),
            value="Config B: LoRA (Raw, r=32)",
            label="Model Configuration",
        ),
    ],
    outputs=[
        gr.Textbox(label="Yoruba Transcription (Unicode NFC)"),
        gr.Textbox(label="Detected Theological Entities (T-NEER)"),
    ],
    title="Neural Oral Bible Translation — Yoruba Whisper + LoRA",
    description=(
        "Compare Zero-Shot Whisper-Small against LoRA-adapted checkpoints on spoken Yoruba scripture "
        "(1.44% trainable parameters, preserving tonal diacritics and underdots)."
    ),
)

if __name__ == "__main__":
    demo.launch()
