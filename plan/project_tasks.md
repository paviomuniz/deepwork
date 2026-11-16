# Project Execution Task Tracker: Neural Oral Bible Translation (ASR)

> **Master Specification:** [project_specification.md](file:///Users/paviomuniz/Library/CloudStorage/OneDrive-Pessoal/Projetos/deepwork/project_specification.md)  
> **Project Plan:** [plan.md](file:///Users/paviomuniz/Library/CloudStorage/OneDrive-Pessoal/Projetos/deepwork/plan/plan.md)  
> **Workflow:** Hybrid Option 1 (Antigravity IDE Local Authoring & Sanity Checks ➔ Google Colab Cloud GPU)

---

## Roadmap Overview

```text
[Phase 1: Architecture & MVP] ──> [Phase 2 & 3: Colab Training & Ablation] ──> [Phase 4: Evaluation & T-NEER] ──> [Phase 5: Final Deliverables]
         (Completed)                             (Completed)                              (Next)                          (Pending)
```

---

## Phase 1: Environment, Modular Architecture & MVP Sanity Check (Completed)

- [x] **Task 1.1: Verify Local Python Environment (`uv` + `.venv`)**
  - **Details:** Confirmed Python 3.11.15 with PyTorch, `transformers`, `peft`, `datasets`, `accelerate`, `evaluate`, `jiwer`, `librosa`, and `soundfile` installed.
- [x] **Task 1.2: Implement Modular `src/` Package**
  - [src/config.py](file:///Users/paviomuniz/Library/CloudStorage/OneDrive-Pessoal/Projetos/deepwork/src/config.py): `ProjectConfig` dataclass (hyperparameters, canonical book splits, curated theological entities) and deterministic `set_seed(42)`.
  - [src/model.py](file:///Users/paviomuniz/Library/CloudStorage/OneDrive-Pessoal/Projetos/deepwork/src/model.py): `openai/whisper-small` loader, PEFT LoRA injection (`q_proj`, `v_proj`, $r=32$, $\alpha=64$, `3,538,944 / 245,273,856` = **1.44%** trainable params), and `apply_spec_augment()` ($F=27, T=100$).
  - [src/dataset.py](file:///Users/paviomuniz/Library/CloudStorage/OneDrive-Pessoal/Projetos/deepwork/src/dataset.py): `Imakandi-Labs/bibletss-yoruba` loader, 16 kHz mono resampling, `[1.0s, 30.0s]` duration filter, strict Unicode NFC tone-preserving normalization, canonical book-level split (`train: 2,927`, `validation: 209` [`1CO`, `EZR`], `test: 151` [`COL`, `GAL`, `EPH`]), and `DataCollatorSpeechSeq2SeqWithPadding`.
  - [src/evaluate.py](file:///Users/paviomuniz/Library/CloudStorage/OneDrive-Pessoal/Projetos/deepwork/src/evaluate.py): Diacritic-preserving & diacritic-stripped WER/CER (`compute_asr_metrics`) and Theological Named Entity Error Audit (`compute_theological_entity_audit`).
  - [src/train.py](file:///Users/paviomuniz/Library/CloudStorage/OneDrive-Pessoal/Projetos/deepwork/src/train.py): `Seq2SeqTrainer` orchestration for Configurations B & C (`label_smoothing_factor=0.0` for Whisper `decoder_input_ids` compatibility).
- [x] **Task 1.3: Phase 1 MVP — 1-Batch Overfit Sanity Check**
  - **Script:** [src/sanity_check.py](file:///Users/paviomuniz/Library/CloudStorage/OneDrive-Pessoal/Projetos/deepwork/src/sanity_check.py)
  - **Verification (Apple Silicon `mps`, `batch_size=2`, gradient checkpointing):** Loss converged cleanly from `4.0805` (Step 1) to `0.0402` (Step 15).
  - **Metrics & SpecAugment Unit Check:** Verified `compute_asr_metrics`, `compute_theological_entity_audit`, and `(2, 80, 3000)` shape preservation.
- [x] **Task 1.4: Create Google Colab Training Notebook**
  - **Deliverable:** [notebooks/01_whisper_lora_training.ipynb](file:///Users/paviomuniz/Library/CloudStorage/OneDrive-Pessoal/Projetos/deepwork/notebooks/01_whisper_lora_training.ipynb) (configured with `torchao` conflict removal, single-process dataset mapping, and dataset reuse across Configurations B & C).

---

## Phase 2 & 3: Colab GPU Execution & 3-Way Ablation Study (Completed)

- [x] **Task 2.1: Run Configuration A — Zero-Shot Baseline (Colab A100)**
  - **Observed Results (Held-out Test Set: `COL`, `GAL`, `EPH`):**
    - `test_loss`: `4.0177`
    - `test_wer`: `113.2419%`
    - `test_cer`: `81.9316%`
- [x] **Task 2.2: Execute Configuration B (`LoRA without SpecAugment`) & Configuration C (`LoRA + SpecAugment`)**
  - **Configuration B (`lora_raw`, 6 epochs, Colab A100) — Held-out Test Set (`COL`, `GAL`, `EPH`):**
    - `test_loss`: `0.6076`
    - `test_wer`: `48.6943%` (down from `113.2419%` — **-64.55 pts**)
    - `test_cer`: `18.0046%` (down from `81.9316%` — **-63.93 pts**)
    - Saved adapter: `checkpoints/whisper-small-yoruba-lora/lora_raw/best_adapter`
  - **Configuration C (`lora_specaug`, 6 epochs, Colab A100) — Held-out Test Set (`COL`, `GAL`, `EPH`):**
    - `test_loss`: `0.6284`
    - `test_wer`: `50.1187%` (down from `113.2419%` — **-63.12 pts**)
    - `test_cer`: **`17.4420%`** (down from `81.9316%` — **-64.49 pts**, best CER across all configs)
    - Saved adapter: `checkpoints/whisper-small-yoruba-lora/lora_specaug/best_adapter`
- [x] **Task 2.3: Persist Trained LoRA Adapters**
  - **Action:** Saved `lora_raw/best_adapter` and `lora_specaug/best_adapter` to Google Drive (`/MyDrive/deepwork_checkpoints/`).

---

## Phase 4: Quantitative Evaluation & Error Analysis (In Progress — Tooling Ready)

- **Evaluation Pipeline:** [src/phase4_eval.py](file:///Users/paviomuniz/Library/CloudStorage/OneDrive-Pessoal/Projetos/deepwork/src/phase4_eval.py) & [notebooks/02_phase4_evaluation.ipynb](file:///Users/paviomuniz/Library/CloudStorage/OneDrive-Pessoal/Projetos/deepwork/notebooks/02_phase4_evaluation.ipynb) (ready for Colab execution against `/MyDrive/deepwork_checkpoints/`).
- [ ] **Task 4.1: Diacritic-Preserving vs. Diacritic-Stripped Breakdown**
  - **Action:** Run `compute_asr_metrics(preds, refs)` on Config A, B, and C test predictions to quantify tone-mark and underdot (`ẹ`, `ọ`, `ṣ`) accuracy gains.
- [ ] **Task 4.2: Theological Named Entity Error Audit (T-NEER)**
  - **Action:** Run `compute_theological_entity_audit(preds, refs)` across the curated biblical proper noun and theological term list in `ProjectConfig.theological_entities`.
- [ ] **Task 4.3: Qualitative Side-by-Side Verse Comparisons**
  - **Action:** Extract 3–5 representative test verses (`COL`, `GAL`, `EPH`) showing Ground Truth vs. Zero-Shot Baseline vs. LoRA + SpecAugment transcripts alongside `IPython.display.Audio` playback.

---

## Phase 5: Final Presentation & Repository Sync (Pending)

- [ ] **Task 5.1: Update Presentation Deck with Empirical Results**
  - **Action:** Populate [slides.md](file:///Users/paviomuniz/Library/CloudStorage/OneDrive-Pessoal/Projetos/deepwork/presentation/slides.md), [index.html](file:///Users/paviomuniz/Library/CloudStorage/OneDrive-Pessoal/Projetos/deepwork/presentation/index.html), and [MultiModal_Audio_Alignment.pptx](file:///Users/paviomuniz/Library/CloudStorage/OneDrive-Pessoal/Projetos/deepwork/presentation/MultiModal_Audio_Alignment.pptx) with final WER/CER/T-NEER tables and qualitative Yoruba diffs.
- [ ] **Task 5.2: Commit & Push Final Artifacts to GitHub**
  - **Action:** Commit `src/`, `notebooks/`, `plan/project_tasks.md`, and updated presentation assets to `origin/main`.

