# Final Project Proposal: Deep Learning Applications in Bible Translation

---

### Project Idea 1

* **Project Title:** Neural Oral Bible Translation: Fine-Tuning Whisper for Low-Resource Scripture Speech Recognition
* **Brief Overview:** This project aims to adapt OpenAI's Whisper model to automatically transcribe spoken biblical audio in an under-resourced Sub-Saharan African language (such as Yoruba or Hausa). By applying Parameter-Efficient Fine-Tuning (LoRA), the system will enable oral-first communities to convert spoken scripture drafts into accurate text with appropriate tonal markings. The goal is to accelerate the quality-checking phase of Oral Bible Translation workflows.
* **Data Requirements:** 
  * [BibleTTS Corpus](https://github.com/BibleTTS/bibletts) (High-fidelity 48kHz studio audio recordings, verse-aligned with official text for African languages, CC BY-SA 4.0).
* **Frameworks/Packages/Libraries Needed:** 
  * `PyTorch`, `Hugging Face Transformers`, `Hugging Face Datasets`, `PEFT` (LoRA), `torchaudio`, `librosa`, `evaluate`, `jiwer`, `accelerate`.
* **Potential Challenges:** Managing high GPU memory consumption during audio spectrogram processing within free Kaggle/Colab tiers, and correctly handling complex tonal diacritics that standard multilingual tokenizers frequently omit.

---

### Project Idea 2

* **Project Title:** Low-Resource Neural Machine Translation of Scripture via Parameter-Efficient Transfer Learning
* **Brief Overview:** This project focuses on fine-tuning Meta's open-source NLLB-200 (No Language Left Behind) model to translate biblical texts from a high-resource source language into an endangered or low-resource target language. Leveraging the parallel structure of scripture, the project aims to explore whether LoRA fine-tuning and back-translation can produce coherent, grammatically sound verse drafts despite severe training data constraints. The objective is to establish an effective machine translation baseline benchmark for minority language communities.
* **Data Requirements:** 
  * [eBible Parallel Corpus](https://github.com/BibleNLP/ebible) (Over 1,000 verse-aligned Bible translations across 800+ languages).
* **Frameworks/Packages/Libraries Needed:** 
  * `PyTorch`, `Hugging Face Transformers`, `Hugging Face Datasets`, `PEFT`, `sacrebleu`, `sentencepiece`, `bitsandbytes`.
* **Potential Challenges:** Mitigating morphological errors and lexical hallucinations caused by the small training corpus, as well as preserving nuanced theological terms across languages with divergent syntactic structures.

---

### Project Idea 3

* **Project Title:** Cross-Lingual Semantic Quality Estimation and Anomaly Detection for Translated Scripture
* **Brief Overview:** This project aims to build a deep learning quality assurance filter that automatically detects omissions, semantic shifts, and mistranslations in newly drafted biblical verses against an anchor reference text. Using a cross-encoder architecture fine-tuned on sentence embeddings (e.g., LaBSE or Sonar), the model will output a semantic equivalence score and flag corrupted or unfaithful translations. This provides automated assistance for human translation teams reviewing community drafts.
* **Data Requirements:** 
  * [eBible Parallel Corpus](https://github.com/BibleNLP/ebible) paired with synthetically generated negative/perturbed verses (introduced through clause deletions, antonym insertions, and entity swaps).
* **Frameworks/Packages/Libraries Needed:** 
  * `PyTorch`, `sentence-transformers`, `Hugging Face Transformers`, `scikit-learn`, `pandas`, `numpy`.
* **Potential Challenges:** Designing realistic synthetic perturbations that accurately reflect human translation errors rather than obvious random noise, and establishing calibrated decision thresholds across languages with varying linguistic distances.
