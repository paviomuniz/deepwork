# Neural Oral Bible Translation: Low-Resource ASR with Whisper & LoRA

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-orange)](https://huggingface.co/)
[![Dataset: BibleTTS](https://img.shields.io/badge/Dataset-BibleTTS-green.svg)](https://github.com/BibleTTS/bibletts)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An applied deep learning project to adapt **OpenAI Whisper** for **Automatic Speech Recognition (ASR)** on low-resource and oral-first languages using the **BibleTTS** corpus.

---

## 📖 Background & Motivation

Over 1 billion people worldwide belong to oral-first or unwritten language communities. In traditional Bible translation, workflows assume literacy and rely heavily on written drafts. Modern initiatives (such as SIL International’s Oral Bible Translation programs) record spoken scripture drafts directly.

However, checking these oral drafts is time-consuming. Standard multilingual speech recognition models (like vanilla Whisper) perform poorly on under-resourced African languages due to:
1. Training data scarcity in web crawls.
2. Complex tone marks and diacritics (e.g., in Yoruba or Ewe) that get omitted or hallucinated.

This project uses **Parameter-Efficient Fine-Tuning (PEFT / LoRA)** to adapt a pretrained Whisper model on high-fidelity studio Bible recordings, creating an accurate transcription copilot that preserves vital orthographic and tonal nuances.

---

## 📂 Repository Structure

```text
├── README.md               # Main project overview and setup guide
├── project_proposal.md     # Academic project proposal (3 candidate ideas)
├── plan/
│   └── plan.md             # Detailed learning plan, milestones, and video tutorials
├── data/                   # (Local cache) Processed audio & metadata (gitignored)
├── notebooks/              # Kaggle / Colab exploration & training notebooks
└── src/                    # Reusable training, preprocessing & evaluation scripts
```

---

## 🎯 Project Highlights

* **Base Architecture:** `openai/whisper-small` or `openai/whisper-base` (Encoder-Decoder Transformer).
* **Adaptation Technique:** **LoRA (Low-Rank Adaptation)** applied to query/value projections, allowing fine-tuning under 8 GB VRAM.
* **Dataset:** [BibleTTS](https://github.com/BibleTTS/bibletts) (48 kHz studio audio, verse-level alignment, Sub-Saharan African languages such as Yoruba, Hausa, or Akuapem Twi).
* **Metrics:** Word Error Rate (WER) and Character Error Rate (CER), with a dedicated audit on biblical proper nouns.

---

## 🛠️ Tech Stack & Requirements

* **Language:** Python 3.10+
* **Deep Learning Framework:** PyTorch
* **Hugging Face Ecosystem:** `transformers`, `datasets[audio]`, `peft`, `accelerate`, `evaluate`
* **Audio Processing:** `torchaudio`, `librosa`, `soundfile`
* **Evaluation:** `jiwer`

---

## 🚀 Quick Start Guide

### 1. Clone & Setup Environment

```bash
git clone https://github.com/your-username/deepwork.git
cd deepwork

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Recommended `requirements.txt`

```text
torch>=2.0.0
torchaudio>=2.0.0
transformers>=4.38.0
datasets[audio]>=2.17.0
peft>=0.9.0
accelerate>=0.27.0
evaluate>=0.4.1
jiwer>=3.0.3
librosa>=0.10.1
soundfile>=0.12.1
```

### 3. Pipeline Flow

```text
1. Load BibleTTS audio (48 kHz) ──> Resample to 16 kHz mono
2. Extract Log-Mel Spectrograms (WhisperFeatureExtractor)
3. Tokenize target verse text (WhisperTokenizer)
4. Train LoRA adapter weights (Seq2SeqTrainer with fp16)
5. Evaluate on held-out Biblical books (WER / CER)
```

---

## 📊 Presentation & Evaluation Strategy

The project culminates in a side-by-side comparative demo that anyone can visually and auditorily inspect:

| Audio Clip | Ground Truth (BibleTTS) | Baseline Whisper (Zero-Shot) | Fine-Tuned Whisper (LoRA) |
| :--- | :--- | :--- | :--- |
| ▶️ *[5s Studio Verse]* | `Nítorí Ọlọ́run fẹ́ aráyé...` | *[Hallucinated English / Failed]* ❌ | `Nítorí Ọlọ́run fẹ́ aráyé...` ✅ |

---

## 📚 References & Acknowledgments

* **BibleTTS:** [OpenSLR / GitHub](https://github.com/BibleTTS/bibletts)
* **BibleNLP Community:** [Awesome Bible NLP](https://github.com/BibleNLP/awesome-bible-nlp)
* **OpenAI Whisper Paper:** *Robust Speech Recognition via Large-Scale Weak Supervision* (Radford et al., 2022)
* **SIL International:** [Scripture Forge & Language Technology](https://www.sil.org)
