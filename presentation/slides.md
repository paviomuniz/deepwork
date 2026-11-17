---
marp: true
theme: default
paginate: true
backgroundColor: #0b0f19
color: #f8fafc
style: |
  section {
    font-family: 'Inter', sans-serif;
    padding: 40px;
    background: radial-gradient(circle at 10% 20%, rgba(30, 41, 59, 0.4) 0%, rgba(11, 15, 25, 0.98) 90%);
  }
  h1, h2, h3 {
    font-family: 'Outfit', sans-serif;
    color: #38bdf8;
  }
  .highlight {
    color: #818cf8;
  }
  .card {
    background: rgba(22, 29, 47, 0.85);
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-radius: 12px;
    padding: 20px;
    margin-top: 15px;
  }
---

# Multi-Modal Audio-Language Alignment
### Bridging Continuous Acoustic Foundation Encoders with Autoregressive LLMs

**Deep Learning Seminar • Special Topic Presentation**  
*Speaker:* [Your Full Name]  
*Target Duration:* 10 to 12 minutes

---

## 1. The Modality Disconnect
### Physical Waves vs. Symbolic Tokens

* **Continuous Physical Acoustics:** $\sim 16,000\text{ samples/sec}$ continuous pressure wave; encodes emotion, prosody, and linguistic tone.
* **Discrete Symbolic Language:** $\sim 3\text{ tokens/sec}$ discrete causal token sequences.
* **The Core Objective:** Project continuous physical representations directly into the latent embedding manifold of autoregressive LLMs without text intermediaries.

---

## 2. Motivation: The Cascade Bottleneck
### Why Chaining ASR ➔ LLM ➔ TTS Fails

* **Error Compounding:** A single misheard phoneme corrupts downstream causal reasoning.
* **Paralinguistic Annihilation:** Text tokenization completely destroys vocal tone, emotion, pitch, sarcasm, and hesitation.
* **Latency Overhead:** Sequential inference across three independent neural pipelines creates unnatural conversational pauses.
* **End-to-End Alignment:** Projects acoustic features directly into LLM prefix embeddings, preserving rich vocal texture.

---

## 3. Representing Sound
### Continuous Feature Maps vs. Discrete Neural Codecs

* **Continuous Foundation Encoders (Whisper / HuBERT / Conformer):**
  * $\mathbf{X} \in \mathbb{R}^{80 \times T} \rightarrow \text{Transformer} \rightarrow \mathbf{H} \in \mathbb{R}^{\frac{T}{2} \times d_{\text{audio}}}$
  * Preserves rich floating-point continuous representations; bridged via neural adapters.
* **Discrete Neural Codecs (EnCodec / SoundStream):**
  * Residual Vector Quantization (RVQ) turns audio into discrete integer tokens.
  * **Sequence Explosion:** $50\text{--}100\text{ frames/sec} \times 8\text{ codebooks} = 400\text{--}800\text{ tokens/sec}$, saturating LLM context windows.

---

## 4. The Alignment Bridge
### 3 Architectural Topologies for Cross-Modal Projection

1. **Linear / MLP Projection Bridge:**
   * $\mathbf{Z} = \text{GELU}(\mathbf{H}\mathbf{W}_1)\mathbf{W}_2$ (maps $d_{\text{audio}} \rightarrow d_{\text{LLM}}$)
   * Computationally lightweight, low latency, preserves full temporal sequence.
2. **Perceiver Resampler / Q-Former:**
   * Fixed learned query latents (e.g., $N=64$) cross-attend to audio feature maps.
   * Compresses variable-length audio into a fixed token budget, protecting context.
3. **Gated Cross-Attention:**
   * Interleaves audio cross-attention layers directly inside the LLM decoder blocks.

---

## 5. Mathematical Objectives
### Two-Stage Training Curriculum

* **Stage 1: Contrastive Feature Alignment (InfoNCE / CLAP):**
  $$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(\mathbf{z}_a, \mathbf{z}_t)/\tau)}{\sum_{j=1}^B \exp(\text{sim}(\mathbf{z}_a, \mathbf{z}_{t,j})/\tau)}$$
* **Stage 2: Multimodal Autoregressive Generation:**
  $$\mathcal{L}_{\text{AR}} = -\sum_{i=1}^N \log P(y_i \mid y_{< i}, \mathbf{E}_{\text{aligned}})$$
* **Preserving Pretrained Knowledge:** Base LLM weights kept frozen or updated via **LoRA** ($W = W_0 + \frac{\alpha}{r}BA$) to prevent catastrophic forgetting.

---

## 6. The 100 Hz Bottleneck
### Solving Quadratic $\mathcal{O}(L^2)$ Attention Saturation

* **The Mismatch:** 30s audio = 3,000 raw feature frames vs. ~80 text tokens (**~35:1 ratio**).
* **1D Strided Convolutions:** Kernels with stride $s=4$ reduce temporal sequence length by 75%.
* **Frame Stacking / Concatenation:** Groups 4 consecutive 20ms frames into an 80ms composite vector: $\mathbb{R}^{T \times d} \rightarrow \mathbb{R}^{\frac{T}{4} \times 4d}$.
* **Result:** Compresses 3,000 frames to ~750 tokens without losing linguistic intelligibility.

---

## 7. Frontier SOTA Implementations
### Empirical Benchmarks & Open Checkpoints

* **Google AudioPaLM:** Unifies ASR, MT, and TTS in a PaLM-2 backbone; outperforms cascaded models on CoVoST-2 speech translation.
* **Meta SeamlessM4T:** Massively multilingual (100+ languages); gains **+3.2 to +5.8 BLEU** over cascaded systems while preserving vocal timbre.
* **Alibaba Qwen2-Audio:** Native conversational voice model; retains **>85% accuracy** under 10dB SNR background noise where ASR cascades fail.

---

## 8. Limitations & Failure Modes
### Modality Collapse, Hallucinations & Compute

* **Modality Collapse (Language Prior Dominance):** The multi-billion parameter text prior ignores weak acoustic adapter signals, outputting plausible text that contradicts the audio.
* **Acoustic Hallucination Loops:** Silent pauses and background static trigger repetitive autoregressive loops in cross-attention layers.
* **Compute Asymmetry & Streaming:** End-to-end multimodal pretraining demands 80GB GPU clusters; chunked real-time streaming requires complex causal masking.

---

## 9. The 12–24 Month Frontier
### From Bolted Adapters to Native Omni-Modality

* **Native Omni-Modality:** Single foundation transformer pretrained from Step 0 on interleaved tokens (GPT-4o, Gemini 2.0).
* **Full-Duplex Conversational Reasoning:** Simultaneous listening and speaking with sub-200ms latency and real-time interruption handling.
* **Linear State-Space Models (Mamba-2):** Replacing quadratic $\mathcal{O}(L^2)$ attention with $\mathcal{O}(N)$ linear-time models for continuous streaming audio.
* **Test-Time Audio Reasoning:** Acoustic chain-of-thought to resolve homophones and acoustic ambiguities before token generation.

---

## 10. Empirical Case Study: BibleTTS Yoruba ASR
### Whisper-Small + LoRA (1.44% Trainable Params) on Held-Out `COL`, `GAL`, `EPH`

| Configuration | WER (%) | WER No-Tone (%) | CER (%) | CER No-Tone (%) | T-NEER F1 (%) |
|---|---:|---:|---:|---:|---:|
| **Config A:** Zero-Shot Baseline | 113.45 | 109.36 | 81.63 | 71.34 | 0.00 |
| **Config B:** LoRA (Raw, $r=32$) | **48.51** | **39.49** | 18.02 | 12.98 | **83.15** |
| **Config C:** LoRA + SpecAugment | 49.96 | 40.23 | **17.44** | **11.60** | 81.36 |

* **-64.94 pts WER / -64.19 pts CER** reduction over Zero-Shot Whisper-Small.
* **T-NEER Audit:** Zero-shot recovers **0 / 104** curated theological entities (`0.00%` F1); LoRA Raw achieves **100.00% Precision** and **83.15% F1** (`74 / 104` TP).
* **Diacritic Bottleneck:** ~9–10 pts of remaining WER (`48.51%` vs. `39.49%`) stems purely from tonal/underdot placement.

---

## 11. Qualitative Analysis: Yoruba Verse Transcripts
### Eliminating Acoustic Hallucination Loops & Recovering Theological Entities

* **`EPH_001_Verse_013` (`Ọlọ́run`, `Olúwa`, `Ẹ̀mí`):**
  * **Ground Truth:** *Ògo ni fún Ọlọ́run àti Baba Jesu Kristi Olúwa wa, ẹni tí ó ti bùkún wa láti inú ọ̀run wá pẹ̀lú àwọn ìbùkún ẹ̀mí gbogbo nínú Kristi.*
  * **Zero-Shot Baseline:** `ḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍ...` *(complete acoustic collapse)*
  * **LoRA (Raw):** *Ìgó ní fún ọlọ́run àti baba Jesu Kristi Olúwa wá, ẹni tó ti bòkún wá láti nú ọ̀run wa pẹ̀lú àwọn ìbòkún ẹmí gbogbo nínú Kristi.*
* **`COL_001_Verse_014` (`Ọlọ́run`, `oore-ọ̀fẹ́`, `Mímọ́`):**
  * **Zero-Shot Baseline:** *Sī yāwenimimo ... āt'yārng dōlōn o bābāwa (×8 repetition loop)*
  * **LoRA (Raw):** *Sí àwọn èní mímọ́ àti àwọn arákùnrin àti arabìnrin ní nú Kristi tí wọn ń gbé ní ìlúkólóse. Orí ọ̀fẹ́ àti àláfíà fún yín láti ọ̀dọ̀ Ọlọ́run baba wà.*

---

## 12. Synthesis & Key Takeaways
### The Multi-Modal Alignment Blueprint

1. **Continuous > Discrete:** Continuous feature maps preserve vocal prosody, pitch, and emotion; RVQ discrete codecs suffer from sequence explosion.
2. **Tame the 100 Hz Mismatch:** Temporal downsampling (1D striding or frame stacking) is mandatory to protect quadratic attention context.
3. **Contrastive + Autoregressive Curriculum:** InfoNCE pre-alignment followed by LoRA fine-tuning prevents catastrophic forgetting.
4. **Universal Paradigm:** The (Encoder $\rightarrow$ Projection Adapter $\rightarrow$ Generative Decoder) pattern is the foundation for all modern multimodal deep learning across audio, vision, and robotics.
