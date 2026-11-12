#!/usr/bin/env python3
"""
Generates the Multi-Modal Audio-Language Alignment PowerPoint presentation (.pptx)
Widescreen 16:9, modern dark obsidian aesthetic, low-text high-signal cards,
dedicated formula callout boxes, architectural process diagrams,
and complete speaker notes embedded in every slide.
"""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

def build_presentation(output_path="presentation/MultiModal_Audio_Alignment.pptx"):
    prs = Presentation()
    # 16:9 Widescreen dimensions: 13.333 x 7.5 inches
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    blank_layout = prs.slide_layouts[6]

    # Color Palette: Modern Obsidian Dark Mode
    BG_COLOR = RGBColor(11, 15, 25)         # Obsidian #0B0F19
    CARD_BG = RGBColor(22, 29, 47)          # Card background #161D2F
    CARD_BORDER = RGBColor(40, 53, 85)      # Card border #283555
    FORMULA_BG = RGBColor(15, 23, 42)       # Formula box background #0F172A
    ACCENT_CYAN = RGBColor(56, 189, 248)    # Cyan #38BDF8
    ACCENT_INDIGO = RGBColor(129, 140, 248) # Indigo #818CF8
    ACCENT_GREEN = RGBColor(52, 211, 153)   # Mint / Emerald #34D399
    ACCENT_ROSE = RGBColor(251, 113, 133)   # Rose / Coral #FB7185
    TEXT_PRIMARY = RGBColor(248, 250, 252)  # Bright White/Slate #F8FAFC
    TEXT_MUTED = RGBColor(148, 163, 184)    # Slate Muted #94A3B8

    def set_slide_background(slide):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
        bg.fill.solid()
        bg.fill.fore_color.rgb = BG_COLOR
        bg.line.fill.background()
        return bg

    def add_header(slide, tag_text, title_text, slide_num, total_slides=10, category="DEEP LEARNING SEMINAR"):
        # Top Category / Tag
        tag_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.45), Inches(9.5), Inches(0.35))
        tf_tag = tag_box.text_frame
        tf_tag.word_wrap = True
        p_tag = tf_tag.paragraphs[0]
        p_tag.text = f"{category.upper()}  •  {tag_text.upper()}"
        p_tag.font.size = Pt(11)
        p_tag.font.bold = True
        p_tag.font.color.rgb = ACCENT_CYAN

        # Slide Number Tracker
        num_box = slide.shapes.add_textbox(Inches(10.5), Inches(0.45), Inches(2.0), Inches(0.35))
        tf_num = num_box.text_frame
        p_num = tf_num.paragraphs[0]
        p_num.text = f"SLIDE {slide_num} / {total_slides}"
        p_num.alignment = PP_ALIGN.RIGHT
        p_num.font.size = Pt(11)
        p_num.font.bold = True
        p_num.font.color.rgb = TEXT_MUTED

        # Main Slide Title
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.8), Inches(11.733), Inches(0.85))
        tf_title = title_box.text_frame
        tf_title.word_wrap = True
        p_title = tf_title.paragraphs[0]
        p_title.text = title_text
        p_title.font.size = Pt(25)
        p_title.font.bold = True
        p_title.font.color.rgb = TEXT_PRIMARY

    def add_card(slide, left, top, width, height, title, content_bullets, border_color=CARD_BORDER, header_color=ACCENT_INDIGO, formula=None):
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        shape.fill.solid()
        shape.fill.fore_color.rgb = CARD_BG
        shape.line.color.rgb = border_color
        shape.line.width = Pt(1.5)

        tf = shape.text_frame
        tf.word_wrap = True
        tf.margin_left = Inches(0.25)
        tf.margin_right = Inches(0.25)
        tf.margin_top = Inches(0.25)
        tf.margin_bottom = Inches(0.25)

        # Title
        p_head = tf.paragraphs[0]
        p_head.text = title
        p_head.font.size = Pt(15)
        p_head.font.bold = True
        p_head.font.color.rgb = header_color
        p_head.space_after = Pt(8)

        # Formula Callout Box (if provided)
        if formula:
            p_form = tf.add_paragraph()
            p_form.text = f"  {formula}  "
            p_form.font.size = Pt(11)
            p_form.font.bold = True
            p_form.font.color.rgb = ACCENT_CYAN
            p_form.space_after = Pt(10)

        # Bullet items
        for bullet in content_bullets:
            p = tf.add_paragraph()
            p.text = f"•  {bullet}"
            p.font.size = Pt(12)
            p.font.color.rgb = TEXT_PRIMARY
            p.space_after = Pt(6)

    def set_speaker_notes(slide, notes_text):
        notes_slide = slide.notes_slide
        tf = notes_slide.notes_text_frame
        tf.text = notes_text

    # =========================================================================
    # SLIDE 1: Title & The Modality Disconnect
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_background(s1)

    # Big Title
    title_box = s1.shapes.add_textbox(Inches(1.0), Inches(1.1), Inches(11.333), Inches(1.9))
    tf1 = title_box.text_frame
    tf1.word_wrap = True
    p1 = tf1.paragraphs[0]
    p1.text = "Multi-Modal Audio-Language Alignment"
    p1.font.size = Pt(36)
    p1.font.bold = True
    p1.font.color.rgb = TEXT_PRIMARY

    p1_sub = tf1.add_paragraph()
    p1_sub.text = "Bridging Acoustic Foundation Encoders with Autoregressive Large Language Models"
    p1_sub.font.size = Pt(20)
    p1_sub.font.color.rgb = ACCENT_CYAN
    p1_sub.space_before = Pt(8)

    p1_meta = tf1.add_paragraph()
    p1_meta.text = "Advanced Deep Learning Seminar • 10–15 Minute Research Presentation"
    p1_meta.font.size = Pt(12)
    p1_meta.font.color.rgb = TEXT_MUTED
    p1_meta.space_before = Pt(6)

    # 2 Comparison Columns
    add_card(s1, Inches(1.0), Inches(3.4), Inches(5.4), Inches(3.3),
             "Continuous Physical Sound",
             [
                 "16,000 samples/sec continuous pressure waveform",
                 "High temporal density, dense acoustic textures",
                 "Encodes emotion, dialect, tone, and speaker identity",
                 "Whisper 80-channel Log-Mel Spectrogram features",
                 "Non-verbal semantics lost in standard ASCII conversion"
             ],
             border_color=ACCENT_INDIGO, header_color=ACCENT_INDIGO,
             formula="Physical Domain: 16 kHz Continuous Audio Wave")

    add_card(s1, Inches(6.9), Inches(3.4), Inches(5.4), Inches(3.3),
             "Discrete Symbolic Language",
             [
                 "~3 tokens/sec discrete causal text sequences",
                 "Sparse temporal density, high symbolic semantics",
                 "Standard BPE token vocabularies (|V| ~ 32k - 128k)",
                 "Autoregressive next-token prediction manifold",
                 "The Goal: Project audio directly into LLM latent space"
             ],
             border_color=ACCENT_CYAN, header_color=ACCENT_CYAN,
             formula="Cognitive Domain: Autoregressive Next-Token Manifold")

    set_speaker_notes(s1,
        "Welcome everyone. Today we explore Multi-Modal Audio-Language Alignment. "
        "Large language models operate in discrete symbolic token spaces. But the physical world speaks "
        "in continuous pressure waves—16,000 variations per second. Today we examine how modern foundation models "
        "eliminate cascaded pipelines and project acoustic representations directly into autoregressive LLM embedding manifolds."
    )

    # =========================================================================
    # SLIDE 2: The Cascade Tax vs. End-to-End Alignment
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    set_slide_background(s2)
    add_header(s2, "Motivation & Problem Definition", "Why Align Modalities? The Cascade Tax & Information Loss", 2)

    add_card(s2, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0),
             "The Traditional Cascade Pipeline",
             [
                 "Pipeline: Audio -> ASR (Whisper) -> Text -> LLM -> Text -> TTS",
                 "Error Compounding: A single misheard word corrupts the entire downstream reasoning context.",
                 "Paralinguistic Annihilation: Pitch, tone, emotion, vocal timbre, and sarcasm are completely destroyed by ASCII tokenization.",
                 "Tonal Collapse: In tonal languages (e.g., Mandarin, Yoruba), stripping acoustic pitch alters lexical word meaning.",
                 "Latency Overhead: Sequential inference overhead across 3 distinct deep neural models."
             ],
             border_color=ACCENT_ROSE, header_color=ACCENT_ROSE,
             formula="Pipeline Bottleneck: Audio -> ASR -> Text -> LLM -> TTS")

    add_card(s2, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.0),
             "Direct End-to-End Alignment",
             [
                 "Unified Latent Space: Direct projection of continuous acoustic encoder vectors into LLM prefix embeddings.",
                 "Preserves Full Acoustic Context: Understands emotion, urgency, vocal inflections, and tonal word meanings.",
                 "Zero Text Bottleneck: Speech-to-reasoning and speech-to-speech translation without error compounding.",
                 "Direct Gradient Flow: Joint end-to-end optimization across modalities.",
                 "Empirical Superiority: +3 to +5 BLEU points over cascaded baselines."
             ],
             border_color=ACCENT_GREEN, header_color=ACCENT_GREEN,
             formula="Unified Alignment: Audio -> Encoder -> Adapter -> Multimodal LLM")

    set_speaker_notes(s2,
        "Why does alignment matter? In a cascaded pipeline, if ASR mishears a proper noun, the downstream LLM hallucinates. "
        "Furthermore, tonal languages rely on pitch to distinguish word meaning; converting to text strips that nuance. "
        "End-to-end alignment preserves the full acoustic texture while enabling direct multimodal reasoning."
    )

    # =========================================================================
    # SLIDE 3: Continuous Encoders vs. Discrete Tokenizers
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    set_slide_background(s3)
    add_header(s3, "Acoustic Foundations", "Representing Sound: Continuous Feature Maps vs. Discrete RVQ Codecs", 3)

    add_card(s3, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0),
             "1. Continuous Encoders (Whisper / HuBERT)",
             [
                 "Input: 80-channel Log-Mel filterbank spectrograms computed every 10ms.",
                 "Floating-Point Vectors: Rich continuous semantic & acoustic representation in R^d.",
                 "Preserves Fine-Grained Nuance: High accuracy on speech recognition & translation without quantization loss.",
                 "Bridge Requirement: Needs a projection adapter (MLP or Cross-Attention) to inject into an LLM.",
                 "Standard Choice: Foundation backbone for AudioPaLM, SeamlessM4T, and Qwen2-Audio."
             ],
             border_color=ACCENT_CYAN, header_color=ACCENT_CYAN,
             formula="Formula: X in R^(80 x T) -> Transformer -> H in R^((T/2) x d_audio)")

    add_card(s3, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.0),
             "2. Discrete Neural Codecs (EnCodec / SoundStream)",
             [
                 "Residual Vector Quantization (RVQ): Converts continuous audio into K parallel codebook streams.",
                 "Discrete Acoustic Tokens: Can be interleaved directly with text tokens in vocabulary.",
                 "Sequence Length Explosion: 50–100 frames/sec x 8 codebooks = 400–800 tokens per second!",
                 "Context Saturation: Rapidly exhausts standard LLM attention context windows.",
                 "Quantization Trade-off: Lower acoustic fidelity than continuous vectors."
             ],
             border_color=ACCENT_INDIGO, header_color=ACCENT_INDIGO,
             formula="Formula: x_wave -> Encoder -> z -> RVQ(K codebooks) -> c in {1,...,V}^K")

    set_speaker_notes(s3,
        "To connect sound to an LLM, we have continuous encoders like Whisper, and discrete neural codecs like Meta's EnCodec. "
        "Continuous encoders preserve rich acoustic representations in floating-point vectors, but require projection bridges. "
        "Discrete codecs turn audio into integers, but create sequence length explosions of 400 to 800 tokens per second. "
        "Most high-accuracy systems choose continuous encoders bridged via adapters."
    )

    # =========================================================================
    # SLIDE 4: Core Architecture: The Alignment Bridge
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    set_slide_background(s4)
    add_header(s4, "Core Architecture", "The Alignment Bridge: 3 Dominant Architectural Paradigms", 4)

    add_card(s4, Inches(0.8), Inches(1.8), Inches(3.7), Inches(5.0),
             "1. Linear / MLP Adapter",
             [
                 "Dimension Match: Maps d_audio (768/1280) -> d_LLM (4096).",
                 "Minimal Overhead: 2-layer MLP introduces almost zero computational latency.",
                 "1:1 Temporal Mapping: Retains full acoustic frame sequence.",
                 "Trade-off: Requires temporal striding to prevent context exhaustion.",
                 "Examples: LLaMA-Omni, Mini-Omni."
             ],
             border_color=CARD_BORDER, header_color=ACCENT_CYAN,
             formula="Z = GELU(H W_1) W_2")

    add_card(s4, Inches(4.8), Inches(1.8), Inches(3.7), Inches(5.0),
             "2. Perceiver / Q-Former",
             [
                 "Learned Latent Queries: Fixed query vectors cross-attend to audio features.",
                 "Fixed Compression: Compresses 30s of audio into 64 fixed prefix tokens.",
                 "Constant Memory: Immune to variable-length audio context spikes.",
                 "Trade-off: May drop brief phonemic or transient details.",
                 "Examples: Flamingo-Audio, BLIP-2 style."
             ],
             border_color=ACCENT_INDIGO, header_color=ACCENT_INDIGO,
             formula="Q_learnable -> Cross-Attn -> H_audio")

    add_card(s4, Inches(8.8), Inches(1.8), Inches(3.7), Inches(5.0),
             "3. Gated Cross-Attention",
             [
                 "Deep Interleaving: Injects cross-attention directly into LLM decoder layers.",
                 "Layer-wise Fusion: Text tokens query acoustic features at every layer.",
                 "High Capacity: Maximum expressiveness for conversational reasoning.",
                 "Trade-off: Higher training parameter and VRAM footprint.",
                 "Examples: SeamlessM4T UnitY decoder."
             ],
             border_color=ACCENT_GREEN, header_color=ACCENT_GREEN,
             formula="Y_l = softmax(Q K_a^T / sqrt(d)) V_a")

    set_speaker_notes(s4,
        "Here are the three dominant bridge paradigms. The Linear MLP adapter projects dimension d_audio to d_LLM. "
        "The Q-Former/Perceiver uses learned queries to compress audio into fixed token counts like 64 tokens. "
        "Gated cross-attention directly interleaves audio attention across all LLM layers. "
        "Pretrained weights are frozen initially while the adapter bridge is trained."
    )

    # =========================================================================
    # SLIDE 5: Mathematical Objectives & Training Loss
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    set_slide_background(s5)
    add_header(s5, "Mathematical Formulations", "Two-Stage Curriculum: Contrastive Pre-Alignment & Autoregression", 5)

    add_card(s5, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0),
             "Stage 1: Contrastive Pre-Alignment (CLAP / InfoNCE)",
             [
                 "Semantic Calibration: Audio embeddings z_a and text embeddings z_t projected into a shared cosine metric space.",
                 "Contrastive Mechanism: Pulls matching (speech, transcript) pairs together; pushes non-matching distractors apart.",
                 "Temperature Scaling: Learnable parameter tau controls logit distribution sharpness.",
                 "Result: Calibrates acoustic and textual coordinate systems before LLM integration."
             ],
             border_color=ACCENT_CYAN, header_color=ACCENT_CYAN,
             formula="L_InfoNCE = -log [ exp(sim(z_a, z_t)/tau) / sum_j exp(sim(z_a, z_tj)/tau) ]")

    add_card(s5, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.0),
             "Stage 2: Multimodal Autoregressive Cross-Entropy",
             [
                 "Acoustic Conditioning: LLM generates discrete tokens conditioned on prefix audio latents E_aligned.",
                 "Causal Masking: Language model retains left-to-right autoregressive reasoning.",
                 "Catastrophic Forgetting Prevention: LLM base weights kept frozen or updated strictly via LoRA (W_0 + alpha/r * B * A).",
                 "Result: Fluent reasoning grounded in acoustic perception."
             ],
             border_color=ACCENT_GREEN, header_color=ACCENT_GREEN,
             formula="L_AR = - sum_i log P(y_i | y_< i, E_aligned)")

    set_speaker_notes(s5,
        "Training follows a two-stage curriculum. Stage 1 uses contrastive InfoNCE loss (like CLAP) "
        "to pull matching audio and text representations together. "
        "Stage 2 optimizes autoregressive cross-entropy, where the LLM predicts text conditioned on the projected audio latents. "
        "Using LoRA during this stage avoids catastrophic forgetting of the LLM's language abilities."
    )

    # =========================================================================
    # SLIDE 6: The 100 Hz Problem: Temporal Downsampling
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    set_slide_background(s6)
    add_header(s6, "Engineering & Efficiency", "The 100 Hz Bottleneck: Solving Quadratic Attention Saturation", 6)

    add_card(s6, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0),
             "The Sequence Length Disparity",
             [
                 "Raw Acoustic Frames: 100 Hz feature extraction = 3,000 frames for 30s of audio.",
                 "Spoken Text Length: 30s speech = ~60 words = ~80 discrete BPE tokens.",
                 "Length Ratio: ~35 : 1 disparity between audio frames and text tokens.",
                 "Attention Scaling: Transformer self-attention scales quadratically: O(L^2). 3,000 audio tokens saturate LLM compute.",
                 "Context Choking: Consumes multi-turn conversation memory rapidly."
             ],
             border_color=ACCENT_ROSE, header_color=ACCENT_ROSE,
             formula="Disparity: 30s Audio = 3,000 frames vs. 80 Text Tokens (~35:1)")

    add_card(s6, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.0),
             "Downsampling Strategies",
             [
                 "1D Strided Convolutions: Kernel k=3, stride s=4 achieves a 75% sequence length reduction.",
                 "Frame Stacking / Concatenation: Stacking 4 adjacent 20ms vectors into an 80ms composite vector.",
                 "CTC Blank Pruning: Removing non-speech and silence frames before LLM injection.",
                 "Outcome: Compresses 3,000 frames to ~750 tokens with zero loss in speech intelligibility.",
                 "Protects LLM throughput and fits within single-GPU VRAM limits."
             ],
             border_color=ACCENT_CYAN, header_color=ACCENT_CYAN,
             formula="Downsampling: [B, T, D] -> Conv1D(k=3, s=4) -> [B, T/4, D']")

    set_speaker_notes(s6,
        "Audio feature extractors produce frames every 10 to 20ms—yielding 3,000 frames for 30 seconds of speech. "
        "Text only requires 80 tokens. That is a 35-to-1 mismatch! "
        "Because attention is O(L^2), feeding raw audio frames destroys throughput. "
        "Downsampling via strided convolutions or frame stacking cuts sequence length by 75%, preserving intelligibility while fitting the LLM context window."
    )

    # =========================================================================
    # SLIDE 7: Real-World SOTA Implementations & Benchmarks
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    set_slide_background(s7)
    add_header(s7, "Empirical Benchmarks", "Frontier Implementations: AudioPaLM, SeamlessM4T & Qwen2-Audio", 7)

    add_card(s7, Inches(0.8), Inches(1.8), Inches(3.7), Inches(5.0),
             "Google AudioPaLM",
             [
                 "Backbone: PaLM-2 + AudioLM discrete token vocabulary.",
                 "Unification: Single model handles speech recognition, translation, and synthesis.",
                 "Empirical Result: Substantially outperformed cascaded Whisper + Google Translate baselines on CoVoST-2.",
                 "Speech-to-Speech: Direct voice transfer without intermediate text."
             ],
             border_color=ACCENT_CYAN, header_color=ACCENT_CYAN,
             formula="PaLM-2 Unified Audio-Text")

    add_card(s7, Inches(4.8), Inches(1.8), Inches(3.7), Inches(5.0),
             "Meta SeamlessM4T",
             [
                 "Scale: Massively multilingual across 100+ input/output languages.",
                 "Expressive Translation: Transfers vocal timbre, pitch, and emotion across languages.",
                 "Benchmark Gain: +3.2 to +5.8 BLEU improvement over cascaded models on CoVoST-2.",
                 "UnitY Architecture: Multi-stage aligned transformer."
             ],
             border_color=ACCENT_INDIGO, header_color=ACCENT_INDIGO,
             formula="+3.2 to +5.8 BLEU on CoVoST-2")

    add_card(s7, Inches(8.8), Inches(1.8), Inches(3.7), Inches(5.0),
             "Alibaba Qwen2-Audio",
             [
                 "Backbone: 24-layer continuous audio encoder + Qwen-7B LLM.",
                 "Conversational Audio: Solves speech Q&A, audio event recognition, and music analysis without text prompts.",
                 "Acoustic Robustness: Retains >85% accuracy under 10dB SNR background noise.",
                 "Open Checkpoints: Available on Hugging Face."
             ],
             border_color=ACCENT_GREEN, header_color=ACCENT_GREEN,
             formula=">85% Accuracy @ 10dB SNR Noise")

    set_speaker_notes(s7,
        "In empirical benchmarks, aligned architectures consistently outperform cascaded pipelines. "
        "AudioPaLM proved that direct audio input beats chaining ASR with Google Translate. "
        "Meta's SeamlessM4T showed 3 to 5 BLEU point gains across 100 languages. "
        "Qwen2-Audio achieves direct voice conversation with high robustness under acoustic noise."
    )

    # =========================================================================
    # SLIDE 8: Limitations & Failure Modes: Modality Collapse
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    set_slide_background(s8)
    add_header(s8, "Limitations & Trade-offs", "Failure Modes: Modality Collapse, Prior Dominance & Hallucinations", 8)

    add_card(s8, Inches(0.8), Inches(1.8), Inches(3.7), Inches(5.0),
             "1. Modality Collapse",
             [
                 "Strong Text Prior: LLM contains billions of parameters trained on trillions of text tokens.",
                 "Overpowering: If audio is noisy or low-resource, the text prior ignores the acoustic adapter.",
                 "Failure: Model generates plausible generic English text that completely contradicts what was spoken.",
                 "Vulnerability: Under-resourced accents and tonal dialects."
             ],
             border_color=ACCENT_ROSE, header_color=ACCENT_ROSE,
             formula="Risk: Text Prior Overpowers Audio")

    add_card(s8, Inches(4.8), Inches(1.8), Inches(3.7), Inches(5.0),
             "2. Acoustic Hallucinations",
             [
                 "Silent Audio Failure: Non-speech background noise triggers autoregressive repetition loops.",
                 "Cross-Attention Drift: Decoders advance text generation without incrementing audio attention position.",
                 "Remedy: Requires strict Voice Activity Detection (VAD) and repetition penalty constraints.",
                 "Common in Whisper on silent audio."
             ],
             border_color=CARD_BORDER, header_color=ACCENT_ROSE,
             formula="Risk: Silence & Static Loops")

    add_card(s8, Inches(8.8), Inches(1.8), Inches(3.7), Inches(5.0),
             "3. Compute Asymmetry",
             [
                 "Training Compute: Pre-aligning cross-attention requires multi-node A100/H100 clusters.",
                 "Inference Latency: Autoregressive decoding cannot start until audio chunk is encoded and downsampled.",
                 "Edge Challenge: High memory footprint for real-time mobile deployment.",
                 "PEFT (LoRA) required for democratization."
             ],
             border_color=CARD_BORDER, header_color=TEXT_MUTED,
             formula="Constraint: High Compute Footprint")

    set_speaker_notes(s8,
        "Multimodal alignment faces several key challenges. First, 'Modality Collapse': the LLM's massive text prior "
        "can overpower the audio adapter, outputting high-probability generic text that contradicts the audio. "
        "Second, silence and noise can trigger hallucination loops. "
        "Third, training requires substantial compute, though PEFT/LoRA helps mitigate adaptation costs."
    )

    # =========================================================================
    # SLIDE 9: The 12–24 Month Frontier: Native Omni-Modality
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    set_slide_background(s9)
    add_header(s9, "Future Trajectory", "The 12–24 Month Frontier: From Bolted Adapters to Native Omni-Modality", 9)

    add_card(s9, Inches(0.8), Inches(1.8), Inches(3.7), Inches(5.0),
             "Native Omni-Modality",
             [
                 "Pretrained from Step 0: Interleaved text tokens, audio frames, and image patches trained in a single transformer.",
                 "No Bolted Adapters: Unified attention heads process acoustic pitch and symbolic reasoning natively (GPT-4o, Gemini 2.0).",
                 "Shared Latent Manifold: Eliminates modality transfer loss entirely.",
                 "End-to-End Multimodality: Single neural network."
             ],
             border_color=ACCENT_CYAN, header_color=ACCENT_CYAN,
             formula="Horizon 1: Unified Token Pretraining")

    add_card(s9, Inches(4.8), Inches(1.8), Inches(3.7), Inches(5.0),
             "Full-Duplex Streaming",
             [
                 "Simultaneous I/O: Listening and speaking in the exact same millisecond.",
                 "Natural Interruption: Sub-200ms latency allows natural human conversational turn-taking.",
                 "State Space Models: Linear Mamba-2 architectures replacing quadratic attention for continuous audio streams.",
                 "Zero Latency Bottleneck: Streaming speech."
             ],
             border_color=ACCENT_INDIGO, header_color=ACCENT_INDIGO,
             formula="Horizon 2: Sub-200ms Turn-Taking")

    add_card(s9, Inches(8.8), Inches(1.8), Inches(3.7), Inches(5.0),
             "Test-Time Audio Reasoning",
             [
                 "Acoustic Chain-of-Thought: Model generates latent acoustic hypotheses before speaking or transcribing.",
                 "Phoneme Disambiguation: Solves tonal homophones in under-resourced languages by searching over acoustic alternatives.",
                 "Verifiable Reasoning: Auditory self-correction.",
                 "Inference-Time Scaling: Reasoning over sound."
             ],
             border_color=ACCENT_GREEN, header_color=ACCENT_GREEN,
             formula="Horizon 3: Acoustic Chain-of-Thought")

    set_speaker_notes(s9,
        "Where is the frontier heading? We are moving from bolted-on adapters to native omni-modal foundation models. "
        "In the next 12 to 24 months, look for native omni-modality pretrained from Step 0 on interleaved tokens; "
        "full-duplex conversational streaming with sub-200ms latency; and test-time acoustic reasoning to resolve tonal ambiguities."
    )

    # =========================================================================
    # SLIDE 10: Conclusion & Key Takeaways
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    set_slide_background(s10)
    add_header(s10, "Summary & Takeaways", "Synthesis: The Future of Auditory Intelligence", 10)

    add_card(s10, Inches(0.8), Inches(1.8), Inches(11.733), Inches(5.0),
             "Core Technical Takeaways for Deep Learning Practitioners",
             [
                 "Acoustic Fidelity > Cascaded Text: Direct projection preserves pitch, tone, and prosody that text tokenization irreversibly destroys.",
                 "The 100 Hz Bottleneck is Solved by Striding: 1D strided convolutions and frame stacking compress the 35:1 audio-to-text sequence mismatch, preventing quadratic attention saturation.",
                 "Two-Stage Alignment Prevents Catastrophic Forgetting: Contrastive InfoNCE pre-alignment followed by autoregressive next-token tuning preserves the LLM's language prior.",
                 "Democratization via PEFT / LoRA: While foundation pretraining requires superclusters, parameter-efficient fine-tuning (LoRA) enables low-resource adaptation on accessible single-GPU hardware.",
                 "Universal Multi-Modal Blueprint: The Encoder -> Adapter -> Generative Decoder paradigm is identical across Audio, Vision (LLaVA), and Robotics (RT-2)."
             ],
             border_color=ACCENT_CYAN, header_color=ACCENT_CYAN,
             formula="Final Synthesis • Audio Foundation Encoders + Autoregressive Language Intelligence")

    set_speaker_notes(s10,
        "To summarize: Speech is a continuous physical wave containing emotion, prosody, and linguistic tone that text cannot capture. "
        "Multi-modal alignment connects auditory perception with symbolic reasoning. "
        "The architectural blueprint we covered today—continuous encoders, temporal compression, and two-stage alignment—is the exact foundation powering modern vision, audio, and omni-modal models. "
        "Thank you! I welcome your questions."
    )

    prs.save(output_path)
    print(f"Successfully generated executive PowerPoint presentation at: {output_path}")

if __name__ == "__main__":
    build_presentation()
