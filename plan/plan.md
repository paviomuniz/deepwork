# Project Plan: Neural Oral Bible Translation (ASR with BibleTTS & Whisper)

## 1. Executive Summary

* **Project Title:** Neural Oral Bible Translation: Fine-Tuning Whisper for Low-Resource Scripture Speech Recognition
* **Task Focus (Option 5A):** Automatic Speech Recognition (ASR) to transcribe oral Bible audio drafts in an under-resourced language.
* **Target Dataset:** [BibleTTS](https://github.com/BibleTTS/bibletts) (High-fidelity 48kHz studio recordings, verse-aligned across 10 Sub-Saharan African languages: e.g., Yoruba, Hausa, or Akuapem Twi).
* **Core Technology:** OpenAI Whisper (`whisper-small` or `whisper-base`) fine-tuned with Parameter-Efficient Fine-Tuning (**PEFT / LoRA**) via Hugging Face `transformers`.
* **Execution Environment:** Kaggle Notebooks (free 30 hrs/week 2x T4 GPU) or Google Colab.
* **Key Deliverables:** 
  1. Trained LoRA adapter weights.
  2. Evaluation benchmark (WER and CER metrics comparing baseline vs. fine-tuned).
  3. Interactive Presentation Slide Deck with audio clips and side-by-side ground truth comparisons.

---

## 2. Complete Timeline & Milestone Roadmap

```
[Phase 1: Foundations] ──> [Phase 2: Data Pipeline] ──> [Phase 3: Model Training] ──> [Phase 4: Evaluation] ──> [Phase 5: Presentation]
     (Weeks 1–2)                  (Days 11–16)                (Days 17–24)                 (Days 25–28)              (Days 29–32)
```

---

## 3. Phase-by-Phase Breakdown

### Phase 1: Knowledge Acquisition & Certifications (Weeks 1–2)
**Goal:** Acquire foundational deep learning concepts, speech transformer architecture intuition, and earn verifiable certifications.

#### 1. Kaggle Learn Certifications (Est. 6–8 hours)
* Complete **[Intro to Deep Learning](https://www.kaggle.com/learn/intro-to-deep-learning)**:
  * Learn linear units, activation functions, loss functions, stochastic gradient descent (SGD/Adam), and dropout.
* Complete **[Natural Language Processing](https://www.kaggle.com/learn/natural-language-processing)**:
  * Learn text tokenization, vocabulary mapping, and sequence modeling.

#### 2. Hugging Face Audio Course Certification (Est. 8–10 hours)
* Complete key modules on the **[Hugging Face Audio Course](https://huggingface.co/learn/audio-course)**:
  * **Unit 1 (Audio Fundamentals):** Sampling rates, digital representations of sound, waveforms, and Log-Mel Spectrograms.
  * **Unit 5 (ASR Architecture):** Encoder-Decoder Transformers vs. CTC networks.
  * **Unit 6 (Fine-Tuning Whisper):** Hands-on tutorial adapting Whisper to a low-resource language.

#### Key Knowledge Checklist:
- [ ] Understand why audio must be resampled to **16 kHz mono** for Whisper.
- [ ] Understand how Log-Mel spectrograms turn raw waves into 80-channel feature maps.
- [ ] Understand **Word Error Rate (WER)** and **Character Error Rate (CER)**.
- [ ] Understand how **LoRA** trains only ~1% of model parameters to fit within 8 GB VRAM.

---

### Phase 2: Environment Setup & Data Pipeline (Days 11–16)
**Goal:** Prepare a clean, reproducible audio processing pipeline on Kaggle.

1. **Platform Setup:**
   * Set up a Kaggle Notebook with GPU accelerator turned on (`T4 x 2` or `P100`).
   * Install dependencies: `transformers`, `datasets[audio]`, `peft`, `torchaudio`, `librosa`, `jiwer`, `evaluate`, `accelerate`.
2. **Target Language Selection:**
   * Choose **one primary language** from BibleTTS:
     * *Option A:* **Yoruba** (Rich tonal system and diacritics; high linguistic interest).
     * *Option B:* **Hausa** (Widely spoken Chadic language in West Africa).
     * *Option C:* **Akuapem Twi** (Niger-Congo language; clean single-speaker studio data).
3. **Data Preprocessing Script:**
   * Load the BibleTTS dataset via Hugging Face `load_dataset("bible-nlp/bibletts", "<language>")`.
   * Resample audio clips from 48 kHz to 16 kHz using `cast_column("audio", Audio(sampling_rate=16000))`.
   * Extract features using `WhisperFeatureExtractor` and tokenize text using `WhisperTokenizer`.
   * Create train/validation/test splits (e.g., Gospel of Luke and Acts for training; Romans for validation; John for test).

---

### Phase 3: Model Development & Experiments (Days 17–24)
**Goal:** Run baseline tests and fine-tune Whisper using parameter-efficient methods.

1. **Establish Zero-Shot Baseline:**
   * Run the vanilla `openai/whisper-small` (or `base`) on your BibleTTS test split without any fine-tuning.
   * Calculate baseline WER and CER. (Notice expected failures: hallucinations, missing tone marks, or English bias).
2. **Configure LoRA Fine-Tuning:**
   ```python
   from peft import LoraConfig, get_peft_model

   config = LoraConfig(
       r=32,
       lora_alpha=64,
       target_modules=["q_proj", "v_proj"],
       lora_dropout=0.05,
       bias="none"
   )
   model = get_peft_model(base_whisper_model, config)
   ```
3. **Training Execution:**
   * Train using Hugging Face `Seq2SeqTrainer` with `fp16=True`, batch size 8–16, and gradient accumulation steps of 2–4.
   * Save checkpoints based on lowest validation loss or validation WER.
4. **Assignment Experiments (Run 2 Comparisons):**
   * *Experiment 1:* **Zero-shot vs. LoRA Fine-Tuned** (Measure the main jump in accuracy).
   * *Experiment 2:* **Impact of Tones & Diacritics** (Evaluate performance on standard text vs. stripped diacritic text).

---

### Phase 4: Quantitative Evaluation & Error Analysis (Days 25–28)
**Goal:** Gather hard metrics and qualitative insights for your report.

1. **Metric Calculation:**
   * Use `evaluate.load("wer")` and `evaluate.load("cer")` to benchmark the test set.
   * Generate an evaluation summary table:
     | Model Configuration | Train Data Size | Validation WER | Test WER | Test CER |
     | :--- | :--- | :--- | :--- | :--- |
     | Zero-Shot Whisper-small | 0 hrs | ~80%+ | ~82% | ~54% |
     | Fine-Tuned Whisper (LoRA) | Full NT (~20 hrs) | **~15%** | **~16%** | **~6%** |
2. **Theological & Named Entity Error Audit:**
   * Search specifically for biblical proper names (*Jésù*, *Bẹ́tílẹ́hẹ́mù*, *Jerúsálẹ́mù*) and key theological terms.
   * Document where the model accurately retained tone marks vs. where it substituted phonetically similar words.

---

### Phase 5: Presentation & Final Deliverables (Days 29–32)
**Goal:** Assemble an engaging slide deck and package reproducible code.

#### Presentation Structure (10–12 Minutes)
1. **Slide 1: Title & The Mission** (Why Oral Bible Translation matters for 1B+ oral-first speakers).
2. **Slide 2: The Challenge of Low-Resource Speech** (Why general models fail on tonal African languages).
3. **Slide 3: Dataset Architecture** (BibleTTS, verse-level alignment, studio audio quality).
4. **Slide 4: Deep Learning Pipeline** (Whisper Encoder-Decoder + LoRA adapter diagram).
5. **Slide 5: Experimental Results** (WER / CER reduction graph).
6. **Slide 6: Live Audio & Side-by-Side Comparison (The Wow Factor):**
   * Embed a 5-second audio clip.
   * Show 3 lines: **Ground Truth Text**, **Baseline Whisper (failed/hallucinated)**, and **Your Fine-Tuned Model (Accurate match with tone marks)**.
7. **Slide 7: Theological Error Analysis** (How well biblical proper nouns were captured).
8. **Slide 8: Ethical AI & Conclusion** (AI as a copilot for human translation teams, inspired by SIL Scripture Forge).

---

## 4. Recommended YouTube Video Tutorials (Concept Visualizations)

To build rapid visual and theoretical intuition, watch these curated video explanations:

### A. Audio Signal Processing & Spectrograms
* **3Blue1Brown:** *[The Fourier Transform, but easily explained](https://www.youtube.com/watch?v=spUNpyF58BY)*  
  *Why watch:* The premier visual explanation of how continuous audio waves decompose into frequencies.
* **Valerio Velardo (The Sound of AI):** *[Mel Spectrograms Explained](https://www.youtube.com/watch?v=4vvXNuQcW_Y)*  
  *Why watch:* Explains how the Mel scale mimics human hearing and converts audio into 2D tensor matrices for neural networks.

### B. Speech Transformers & OpenAI Whisper
* **3Blue1Brown:** *[Attention in Transformers, Visually Explained](https://www.youtube.com/watch?v=eMlx5fFNoYc)*  
  *Why watch:* Visualizes the self-attention and cross-attention mechanisms used inside Whisper’s encoder-decoder stack.
* **Yannic Kilcher:** *[Whisper: Robust Speech Recognition via Large-Scale Weak Supervision](https://www.youtube.com/watch?v=ocjYkS_fO0U)*  
  *Why watch:* Clear paper walkthrough detailing Whisper's 30-second window framing, multitasking tokens, and multilingual training.
* **Sanchit Gandhi (Hugging Face):** *[Fine-Tuning Whisper with Hugging Face Transformers](https://www.youtube.com/watch?v=f_Vp_qZ3hE8)*  
  *Why watch:* The official step-by-step code tutorial walking through dataset mapping, feature extractors, and `Seq2SeqTrainer`.

### C. Parameter-Efficient Fine-Tuning (LoRA & PEFT)
* **StatQuest with Josh Starmer:** *[LoRA: Low-Rank Adaptation of LLMs, Clearly Explained!!!](https://www.youtube.com/watch?v=dA-NhCtrrVE)*  
  *Why watch:* Intuitive mathematical explanation of matrix decomposition ($W + A \times B$) that lets you train Whisper on a single 8GB/16GB GPU.
* **Umar Jamil:** *[LoRA: Low-Rank Adaptation from Scratch](https://www.youtube.com/watch?v=PXER4tBduOU)*  
  *Why watch:* Deep code and math walkthrough of low-rank matrices and LoRA alpha scaling.

---

## 5. Curated Resource Directory

| Resource Name | Type | URL |
| :--- | :--- | :--- |
| **Hugging Face Audio Course** | Course & Certificate | [huggingface.co/learn/audio-course](https://huggingface.co/learn/audio-course) |
| **Kaggle Intro to Deep Learning** | Micro-course & Certificate | [kaggle.com/learn/intro-to-deep-learning](https://www.kaggle.com/learn/intro-to-deep-learning) |
| **Kaggle NLP Course** | Micro-course & Certificate | [kaggle.com/learn/natural-language-processing](https://www.kaggle.com/learn/natural-language-processing) |
| **Whisper LoRA Starter Notebook** | Kaggle Code Reference | [kaggle.com/code/nbroad/whisper-training-starter-kit](https://www.kaggle.com/code/nbroad/whisper-training-starter-kit) |
| **BibleTTS Repository** | Dataset & Documentation | [github.com/BibleTTS/bibletts](https://github.com/BibleTTS/bibletts) |
| **BibleNLP Ecosystem** | Community & Papers | [github.com/BibleNLP/awesome-bible-nlp](https://github.com/BibleNLP/awesome-bible-nlp) |

