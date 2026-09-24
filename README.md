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

## 🛠️ Tech Stack & Tooling

* **Package & Environment Manager:** **`uv`** (Astral's ultra-fast Python package and project manager)
* **Development Environment:** **Antigravity IDE** (AI pair programming, code authoring, Git version control)
* **Training Hardware:** **Google Colab Cloud GPU** (Free NVIDIA Tesla T4 16GB) / Kaggle
* **Deep Learning Framework:** PyTorch 2.1+
* **Hugging Face Ecosystem:** `transformers`, `datasets[audio]`, `peft`, `accelerate`, `evaluate`
* **Audio Tooling:** `torchaudio`, `librosa`, `soundfile`
* **Evaluation:** `jiwer`

---

## 🔄 Development Workflow (Option 1: Antigravity ➔ Google Colab)

We follow the **Option 1 hybrid workflow**:

1. **Local Authoring & Architecture in Antigravity IDE:**
   * Write, refactor, and manage modular scripts (`src/`) and notebooks (`notebooks/`) locally with Antigravity’s AI assistant.
   * Run quick sanity checks (Phase 1 MVP 1-batch tests) locally or via local CPU/MPS.
   * Push clean commits to GitHub.
2. **Heavy GPU Execution on Google Colab:**
   * Open the repository's training notebook directly in Google Colab with 1 click:
     [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)
   * Leverage Colab's free cloud NVIDIA T4 GPU for full-scale training on the New Testament (~15 hours of audio).
   * Save fine-tuned LoRA checkpoints to Google Drive or Hugging Face Hub.

---

## 🚀 Quick Start Guide (Powered by `uv`)

### 1. Install `uv` (if not already installed)

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Local Environment Setup

```bash
cd /Users/paviomuniz/Library/CloudStorage/OneDrive-Pessoal/Projetos/deepwork

# Create virtual environment with uv
uv venv

# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install all dependencies in seconds using uv
uv pip install -r requirements.txt
```

### 3. Launching Notebooks in Antigravity or Jupyter

```bash
# Run Jupyter within the uv-managed environment
uv run jupyter lab
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
