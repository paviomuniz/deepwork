# Technical Project Specification: Neural Oral Bible Translation (ASR)

This specification defines the architectural, data, algorithmic, and engineering standards for the project: **Fine-Tuning Whisper for Low-Resource Scripture Speech Recognition via LoRA**.

---

## 1. Problem Definition & Task Type

### 1.1 The Problem
Over 1 billion people speak languages that are predominantly oral or lack established orthographic traditions. In modern Oral Bible Translation (OBT) workflows (e.g., supported by SIL International and Seed Company), translation teams draft and iterate on scripture passages via voice recordings rather than written text. However, reviewing, checking, and aligning these audio drafts is heavily manual and slow. Standard multilingual foundation models (e.g., off-the-shelf OpenAI Whisper) fail significantly on under-resourced African languages due to severe training data scarcity and the omission of critical tone marks and diacritics.

### 1.2 Machine Learning Task Category
* **Task Type:** Supervised Sequence-to-Sequence (Seq2Seq) Speech-to-Text Generation / Automatic Speech Recognition (ASR).
* **Paradigm:** Transfer learning with Parameter-Efficient Fine-Tuning (PEFT / LoRA).

### 1.3 Input & Output Specifications
* **Raw Input:** Variable-length single-channel audio waveform sampled at $16\text{ kHz}$ ($\text{audio} \in \mathbb{R}^{T}$ where duration $1.0\text{s} \le t \le 30.0\text{s}$). Converted into an **80-channel Log-Mel Spectrogram** matrix of shape:
  $$\mathbf{X} \in \mathbb{R}^{80 \times 3000}$$
  (Whisper standard 30-second padded window, 25ms window size, 10ms hop size).
* **Expected Output:** Autoregressive sequence of discrete subword tokens decoded into UTF-8 text representing the verse transcription with tone diacritics:
  $$\mathbf{Y} = (y_1, y_2, \dots, y_U), \quad y_u \in \mathcal{V}$$
  where $\mathcal{V}$ is the Whisper multilingual BPE vocabulary ($|\mathcal{V}| \approx 51,865$).

---

## 2. Dataset & Pipeline Architecture

### 2.1 Data Source
* **Primary Dataset:** **[BibleTTS Corpus](https://github.com/BibleTTS/bibletts)** (Biblica / OpenSLR / Hugging Face `bible-nlp/bibletts`).
* **Characteristics:** Studio-quality 48 kHz single-speaker recordings, canonical verse-level segmentation, verified written scripture transcripts.
* **Target Language:** **Yoruba (`yor`)** exclusively (New Testament portion, ~18–20 hours of high-fidelity studio audio). High linguistic and academic value due to rich tonal diacritics (acute, grave, macron) and underdots (ẹ, ọ, ṣ) that dictate lexical meaning.

### 2.2 Splitting Strategy (Data Leakage & Contamination Prevention)
> [!IMPORTANT]
> **Canonical Group-Based Split (No Random Verse Shuffling):**  
> Never split verses randomly across chapters. Because nearby verses share repetitive contextual vocabulary and syntactic rhythms, random splitting leads to synthetic data leakage.
* **Train Set (~75%, ~15 hours):** Gospels of Matthew, Mark, Luke, and Acts through Revelation (excluding test and validation books).
* **Validation Set (~12.5%, ~2.5 hours):** Epistle to the Romans & 1 Corinthians (used for checkpointing, early stopping, and hyperparameter tuning).
* **Test Set (~12.5%, ~2.5 hours):** Gospel of John (held-out narrative text containing high frequencies of core theological terms and named entities).

### 2.3 Preprocessing & Audio Transformations
1. **Audio Standardization:**
   * Resample from 48 kHz to 16 kHz mono using `torchaudio.transforms.Resample(orig_freq=48000, new_freq=16000)`.
   * Dynamic range normalization to $[-1.0, 1.0]$.
   * Filter out corrupt audio: discard clips with duration $< 1.0\text{s}$ or $> 30.0\text{s}$.
2. **Text Normalization & Diacritic Preservation:**
   * Strip non-speech metadata annotations (chapter headers, reader cues).
   * **Strict Tone Preservation Policy:** Enforce Unicode **NFC** normalization (`unicodedata.normalize('NFC', text)`). Never strip tonal accents or underdots, as tone shifts in Yoruba alter word meanings (e.g., *ọwọ́* = hand vs. *owó* = money).
3. **Feature Extraction:**
   * `WhisperFeatureExtractor` computes 80-bin log-mel filterbanks, normalized to mean 0, variance 1.
4. **Data Augmentation (Anti-Overfitting on Single Studio Voice):**
   * **SpecAugment:** Frequency masking ($F=27$) and Time masking ($T=100$) applied to the log-mel features during training to prevent the model from memorizing the acoustic profile of the single BibleTTS narrator.

---

## 3. Model Architecture & Training Strategy

### 3.1 Baseline Model
* **Zero-Shot Baseline:** Pretrained `openai/whisper-small` (244M parameters) evaluated directly on the held-out Gospel of John test set with target language set (`language='yo'`, `task='transcribe'`) without weight updates.

### 3.2 Core Model Architecture
* **Backbone:** **`openai/whisper-small`** (244M parameters).
  * Audio Encoder: 12-layer Transformer encoder processing audio representations.
  * Text Decoder: 12-layer autoregressive Transformer decoder with cross-attention over encoder states.
* **Adaptation Mechanism:** **LoRA (Low-Rank Adaptation via `peft`)**
  * Target Modules: Query and Value projection layers (`q_proj`, `v_proj`) in both encoder and decoder attention blocks.
  * Rank: $r = 32$, Scaling factor: $\alpha = 64$, Dropout: $0.05$.
  * Trainable Parameters: $\approx 1.2\%$ of total model weights (~3.5M params), keeping peak VRAM $< 10\text{ GB}$ on a Kaggle/Colab T4 GPU.

### 3.3 Loss Function & Optimization Details
* **Loss Function:** Label-Smoothed Cross-Entropy Loss ($\epsilon = 0.1$) computed over target token sequences (ignoring padding tokens with index $-100$):
  $$\mathcal{L}_{\text{CE}} = - \sum_{u=1}^U \log P(y_u \mid y_{<u}, \mathbf{X})$$
* **Optimizer:** `AdamW` ($\beta_1=0.9, \beta_2=0.98, \epsilon=10^{-8}$, weight decay $= 0.01$).
* **Learning Rate Schedule:** Linear warm-up for 500 steps followed by cosine annealing decay down to $10\%$ of peak LR ($1 \times 10^{-4}$).
* **Precision:** Mixed precision `fp16` enabled with dynamic loss scaling.

### 3.4 Staged Execution Strategy
* **Phase 1 (MVP Pipeline / Gradient Sanity Check):**
  * Extract 1 single batch (8 verses). Overfit the model on this single batch for 50 steps until training loss approaches $\approx 0$.
  * Verify gradient backpropagation flows properly through LoRA adapters while keeping base transformer weights frozen (`requires_grad = False`).
* **Phase 2 (Scaled Training & 3-Way Ablation Study):**
  * Train across the full New Testament training split (~15 hours) for 5–8 epochs (effective batch size $= 32$).
  * Conduct the designated **3-Way Comparative Experiment**:
    1. **Configuration A (Zero-Shot Baseline):** Vanilla `whisper-small` without fine-tuning.
    2. **Configuration B (LoRA without SpecAugment):** Fine-tuned Whisper-small with LoRA on raw log-mel features.
    3. **Configuration C (LoRA + SpecAugment):** Fine-tuned Whisper-small with LoRA + frequency and time masking.

---

## 4. Hardware, Environment & Evaluation

### 4.1 Compute Target & Hybrid Workflow (Option 1)
* **Development Workflow:** **Option 1 (Antigravity IDE ➔ Google Colab)**
  * **Local Stage (Antigravity IDE):** Code authoring, refactoring, Git versioning, and Phase 1 MVP (1-batch sanity check on local CPU/MPS).
  * **Training Stage (Google Colab Cloud GPU):** Full New Testament LoRA fine-tuning and ablation runs executed on free NVIDIA Tesla T4 16GB GPUs via Google Colab.
* **VRAM Budget:** Maximum allowed peak memory $\le 10\text{ GB}$ (leaving $\ge 5\text{ GB}$ safety buffer against Out-of-Memory crashes).

### 4.2 Technology Stack & Package Management
* **Python Environment Manager:** **`uv`** (Astral) — used for virtual environment creation (`uv venv`) and lightning-fast package resolution (`uv pip install`).
* Python: `3.10+`
* PyTorch: `2.1.0+`
* Hugging Face: `transformers >= 4.38.0`, `datasets[audio] >= 2.17.0`, `peft >= 0.9.0`, `accelerate >= 0.27.0`
* Audio Tooling: `torchaudio >= 2.1.0`, `librosa >= 0.10.1`, `soundfile >= 0.12.1`
* Evaluation: `jiwer >= 3.0.3`, `evaluate >= 0.4.1`


### 4.3 Evaluation Metrics
1. **Primary Metric:** **Word Error Rate (WER)**
   $$\text{WER} = \frac{S + D + I}{N}$$
   (where $S$ = substitutions, $D$ = deletions, $I$ = insertions, $N$ = total reference words).
2. **Secondary Metric:** **Character Error Rate (CER)**  
   *Essential for Yoruba, as CER captures granular accuracy on tonal diacritics and subword morphology.*
3. **Theological Named Entity Error Audit (T-NEER):**  
   *Dedicated precision and recall tracking on a curated list of ~50 key biblical proper names (e.g., Jésù, Jerúsálẹ́mù, Bẹ́tílẹ́hẹ́mù, Pọ́ọ̀lù, Ábúráhámù) and theological terms (ìgbàlà, òdodo, ìyè).*

### 4.4 Demonstration & Deliverable Format
* **Interactive Kaggle/Colab Notebook:** Clean end-to-end execution notebook with `IPython.display.Audio` players displaying side-by-side transcripts.
* **Audio-Embedded Slide Deck:** Presentation deck featuring audio snippets and color-coded text diffs for class presentation.


---

## 5. AI Guardrails & Code Standards (Deep Learning "Vibe Rules")

To guarantee clean, bug-free implementations, the following strict engineering rules apply:

### 5.1 Tensor Shape Hygiene
* **Explicit Assertions & Comments:** Every major transformation function must include explicit shape annotations and sanity checks:
  ```python
  # Input log-mel features: [batch_size, n_mels=80, time_steps=3000]
  assert mel.shape[1] == 80, f"Expected 80 mel channels, got {mel.shape[1]}"
  ```

### 5.2 Strict Leakage Prevention
* Audio normalizers, tokenizers, and text cleaners must be fitted or validated **only** on the training split.
* Train/Validation/Test data loaders must draw from independent non-overlapping book indices.

### 5.3 Deterministic Reproducibility
* All scripts must invoke a centralized seeding routine at their entry point:
  ```python
  def set_seed(seed: int = 42):
      import random, os, numpy as np, torch
      random.seed(seed)
      os.environ['PYTHONHASHSEED'] = str(seed)
      np.random.seed(seed)
      torch.manual_seed(seed)
      torch.cuda.manual_seed_all(seed)
      torch.backends.cudnn.deterministic = True
  ```

### 5.4 Audio & Speech Domain Guardrails (Enhanced Rules)
* **Padding & Masking Consistency:** Ensure attention masks correctly mask out padded audio frames and padded text label tokens (`labels[labels == tokenizer.pad_token_id] = -100`).
* **VRAM Safety:** Never run unbatched Whisper inference on raw Python loops. Always use a dedicated PyTorch `DataLoader` with pin memory and pre-set batch sizes.
* **Audio Length Safeguard:** Reject or clamp audio longer than $30.0\text{s}$ before inputting into the Whisper encoder to prevent silent sequence truncation errors.

### 5.5 Modularity & Directory Structure
Source code must be strictly decoupled across standalone modules:
```text
src/
├── config.py       # Dataclass with all hyperparams, paths, and seed configs
├── dataset.py      # BibleTTS loading, 16kHz resampling, and collator
├── model.py        # Base Whisper loading, LoRA injection, and parameter freezing
├── train.py        # Hugging Face Seq2SeqTrainer orchestration and checkpointing
└── evaluate.py     # WER, CER, and T-NEER calculation scripts
```
