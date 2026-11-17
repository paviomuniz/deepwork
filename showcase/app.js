(() => {
  const data = window.PHASE4_DATA;
  if (!data) return;

  // --- 1. Accessible Tabs ---
  const tabs = Array.from(document.querySelectorAll('[role="tab"]'));
  const panels = Array.from(document.querySelectorAll('[role="tabpanel"]'));

  function selectTab(targetTab) {
    tabs.forEach((tab, idx) => {
      const active = tab === targetTab;
      tab.setAttribute("aria-selected", String(active));
      const panel = panels[idx];
      if (active) {
        panel.removeAttribute("hidden");
        panel.setAttribute("tabindex", "0");
      } else {
        panel.setAttribute("hidden", "");
        panel.removeAttribute("tabindex");
      }
    });
  }

  tabs.forEach((tab, idx) => {
    tab.addEventListener("click", () => selectTab(tab));
    tab.addEventListener("keydown", (e) => {
      let next = idx;
      if (e.key === "ArrowRight") next = (idx + 1) % tabs.length;
      else if (e.key === "ArrowLeft") next = (idx - 1 + tabs.length) % tabs.length;
      else if (e.key === "Home") next = 0;
      else if (e.key === "End") next = tabs.length - 1;
      else return;
      e.preventDefault();
      tabs[next].focus();
      selectTab(tabs[next]);
    });
  });

  // --- 2. Pipeline Explorer ---
  const pipeDetails = {
    audio: {
      title: "1. Studio Audio Input (48 kHz -> 16 kHz Mono)",
      body: "BibleTTS Yoruba studio recordings (48 kHz) are resampled to 16 kHz mono and filtered to [1.0s, 30.0s] to fit Whisper's 30-second acoustic window."
    },
    mel: {
      title: "2. 80-Channel Log-Mel Spectrogram [B, 80, 3000]",
      body: "WhisperFeatureExtractor converts raw waveforms into 80 log-mel frequency bins across 3,000 frames (10 ms stride) while preserving pitch contours critical for Yoruba tone."
    },
    specaug: {
      title: "3. On-the-Fly SpecAugment (F=27, T=100)",
      body: "Config C applies FrequencyMasking(27) and TimeMasking(100) dynamically in DataCollatorSpeechSeq2SeqWithPadding during training only, reducing CER to 17.44%."
    },
    lora: {
      title: "4. Frozen Whisper-Small + LoRA (r=32, α=64)",
      body: "Base weights stay frozen; low-rank adapters on q_proj and v_proj train just 3,538,944 / 245,273,856 parameters (1.44%). label_smoothing_factor=0.0 ensures Whisper decoder_input_ids compatibility."
    },
    nfc: {
      title: "5. Unicode NFC Tone-Preserving Decoding",
      body: "Strict Unicode NFC normalization preserves Yoruba tonal diacritics (acute/grave) and sub-dots (ẹ, ọ, ṣ), unlike standard ASCII-stripping ASR pipelines."
    }
  };

  const pipeNodes = Array.from(document.querySelectorAll(".pipe-node"));
  const pipeBox = document.getElementById("pipeline-detail");

  function showPipe(step) {
    pipeNodes.forEach((n) => n.setAttribute("aria-pressed", String(n.dataset.step === step)));
    const info = pipeDetails[step];
    pipeBox.innerHTML = `<strong>${info.title}</strong><p style="margin:0.4rem 0 0;">${info.body}</p>`;
  }
  pipeNodes.forEach((n) => n.addEventListener("click", () => showPipe(n.dataset.step)));
  showPipe("audio");

  // --- 3. Interactive SpecAugment Canvas ---
  const canvas = document.getElementById("spec-canvas");
  const ctx = canvas.getContext("2d");
  const btnFreq = document.getElementById("toggle-freq");
  const btnTime = document.getElementById("toggle-time");
  const btnReroll = document.getElementById("reroll-spec");

  let maskFreq = true;
  let maskTime = true;
  let fStart = 42;
  let tStart = 210;

  function drawSpectrogram() {
    const w = canvas.width;
    const h = canvas.height;
    ctx.fillStyle = "#090d16";
    ctx.fillRect(0, 0, w, h);

    // Draw synthetic harmonic speech bands
    for (let x = 0; x < w; x += 4) {
      const energy = 0.4 + 0.6 * Math.abs(Math.sin(x * 0.03) * Math.cos(x * 0.013));
      for (let y = 0; y < h; y += 4) {
        const band = Math.exp(-Math.pow((y - h * 0.68) / 36, 2)) + 0.6 * Math.exp(-Math.pow((y - h * 0.38) / 28, 2));
        const val = Math.min(1, energy * band + 0.08 * Math.sin(x * 0.2 + y * 0.3));
        const r = Math.floor(15 + 220 * val);
        const g = Math.floor(25 + 180 * Math.pow(val, 1.3));
        const b = Math.floor(60 + 195 * Math.sqrt(val));
        ctx.fillStyle = `rgb(${r},${g},${b})`;
        ctx.fillRect(x, y, 4, 4);
      }
    }

    if (maskFreq) {
      const fHeight = Math.round((27 / 80) * h);
      ctx.fillStyle = "rgba(15, 23, 42, 0.88)";
      ctx.fillRect(0, fStart, w, fHeight);
      ctx.strokeStyle = "#38bdf8";
      ctx.strokeRect(0.5, fStart + 0.5, w - 1, fHeight - 1);
      ctx.fillStyle = "#38bdf8";
      ctx.font = "bold 12px Inter, sans-serif";
      ctx.fillText("Freq Mask (F=27 bins)", 10, fStart + 18);
    }

    if (maskTime) {
      const tWidth = Math.round((100 / 600) * w);
      ctx.fillStyle = "rgba(15, 23, 42, 0.88)";
      ctx.fillRect(tStart, 0, tWidth, h);
      ctx.strokeStyle = "#34d399";
      ctx.strokeRect(tStart + 0.5, 0.5, tWidth - 1, h - 1);
      ctx.fillStyle = "#34d399";
      ctx.font = "bold 12px Inter, sans-serif";
      ctx.fillText("Time Mask (T=100)", tStart + 8, 20);
    }
  }

  btnFreq.addEventListener("click", () => {
    maskFreq = !maskFreq;
    btnFreq.setAttribute("aria-pressed", String(maskFreq));
    drawSpectrogram();
  });
  btnTime.addEventListener("click", () => {
    maskTime = !maskTime;
    btnTime.setAttribute("aria-pressed", String(maskTime));
    drawSpectrogram();
  });
  btnReroll.addEventListener("click", () => {
    fStart = 15 + Math.floor(Math.random() * 110);
    tStart = 40 + Math.floor(Math.random() * 430);
    drawSpectrogram();
  });
  drawSpectrogram();

  // --- 4. 3-Way Ablation Dashboard ---
  const configs = [
    { key: "config_a_baseline", label: "Config A: Zero-Shot", color: "var(--rose)" },
    { key: "config_b_lora_raw", label: "Config B: LoRA (Raw)", color: "var(--accent)" },
    { key: "config_c_lora_specaug", label: "Config C: LoRA + SpecAug", color: "var(--emerald)" }
  ];

  const btnTone = document.getElementById("mode-tone");
  const btnNoTone = document.getElementById("mode-notone");
  const werBars = document.getElementById("wer-bars");
  const cerBars = document.getElementById("cer-bars");

  function renderAblationBars(noTone) {
    btnTone.setAttribute("aria-pressed", String(!noTone));
    btnNoTone.setAttribute("aria-pressed", String(noTone));

    const werField = noTone ? "wer_no_tone" : "wer";
    const cerField = noTone ? "cer_no_tone" : "cer";

    werBars.innerHTML = configs
      .map((c) => {
        const val = data.metrics[c.key][werField];
        const pct = Math.min(100, (val / 120) * 100).toFixed(1);
        return `<div class="bar-row">
          <span>${c.label}</span>
          <div class="bar-track"><div class="bar-fill" style="width:${pct}%;background:${c.color}"></div></div>
          <strong>${val.toFixed(2)}%</strong>
        </div>`;
      })
      .join("");

    cerBars.innerHTML = configs
      .map((c) => {
        const val = data.metrics[c.key][cerField];
        const pct = Math.min(100, (val / 90) * 100).toFixed(1);
        return `<div class="bar-row">
          <span>${c.label}</span>
          <div class="bar-track"><div class="bar-fill" style="width:${pct}%;background:${c.color}"></div></div>
          <strong>${val.toFixed(2)}%</strong>
        </div>`;
      })
      .join("");
  }

  btnTone.addEventListener("click", () => renderAblationBars(false));
  btnNoTone.addEventListener("click", () => renderAblationBars(true));
  renderAblationBars(false);

  document.getElementById("ablation-tbody").innerHTML = configs
    .map((c) => {
      const m = data.metrics[c.key];
      return `<tr>
        <th>${c.label}</th>
        <td data-label="WER (%)">${m.wer.toFixed(2)}</td>
        <td data-label="WER No-Tone (%)">${m.wer_no_tone.toFixed(2)}</td>
        <td data-label="Diacritic WER Gap">+${m.wer_diacritic_gap.toFixed(2)}</td>
        <td data-label="CER (%)">${m.cer.toFixed(2)}</td>
        <td data-label="CER No-Tone (%)">${m.cer_no_tone.toFixed(2)}</td>
        <td data-label="Diacritic CER Gap">+${m.cer_diacritic_gap.toFixed(2)}</td>
      </tr>`;
    })
    .join("");

  // --- 5. T-NEER Audit ---
  document.getElementById("tneer-bars").innerHTML = configs
    .map((c) => {
      const m = data.metrics[c.key];
      return `<div class="bar-row">
        <span>${c.label}</span>
        <div class="bar-track"><div class="bar-fill" style="width:${m.t_neer_f1}%;background:${c.color}"></div></div>
        <strong>${m.t_neer_f1.toFixed(2)}% F1</strong>
      </div>
      <div style="font-size:0.82rem;color:var(--muted);margin:-0.4rem 0 0.4rem;">
        Precision: ${m.t_neer_precision.toFixed(2)}% · Recall: ${m.t_neer_recall.toFixed(2)}% · TP: ${m.t_neer_tp} / ${m.t_neer_tp + m.t_neer_fn}
      </div>`;
    })
    .join("");

  document.getElementById("entity-grid").innerHTML = data.theological_entities
    .map((e) => `<span class="entity-pill">${e}</span>`)
    .join("");

  // --- 6. Yoruba Verse Diff Explorer ---
  const verseSelect = document.getElementById("verse-select");
  const verseSearch = document.getElementById("verse-search");
  const verseContainer = document.getElementById("verse-diff-container");

  const stripDiacritics = (s) =>
    s.normalize("NFD").replace(/[\u0300-\u036f]/g, "").normalize("NFC").toLowerCase();

  function escapeHtml(str) {
    return str.replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
  }

  function highlightDiff(pred, ref) {
    const refWords = ref.trim().split(/\s+/);
    const refNormSet = new Set(refWords.map((w) => w.toLowerCase().replace(/[.,;:!?]/g, "")));
    const refNoToneSet = new Set(refWords.map((w) => stripDiacritics(w).replace(/[.,;:!?]/g, "")));

    return pred
      .trim()
      .split(/\s+/)
      .map((tok) => {
        const clean = tok.toLowerCase().replace(/[.,;:!?]/g, "");
        const cleanNoTone = stripDiacritics(tok).replace(/[.,;:!?]/g, "");
        const safe = escapeHtml(tok);
        if (refNormSet.has(clean)) return `<span class="tok-ok">${safe}</span>`;
        if (refNoToneSet.has(cleanNoTone)) return `<span class="tok-tone">${safe}</span>`;
        return `<span class="tok-err">${safe}</span>`;
      })
      .join(" ");
  }

  const qualByIndex = new Map(data.qualitative_samples.map((s) => [s.index, s]));

  function populateSelect(filterText = "") {
    const q = filterText.trim().toLowerCase();
    const filtered = data.verses.filter(
      (v) => !q || v.ref.toLowerCase().includes(q) || stripDiacritics(v.ref).includes(stripDiacritics(q))
    );
    verseSelect.innerHTML = filtered
      .map((v) => {
        const qual = qualByIndex.get(v.index);
        const label = qual ? `★ ${qual.verse_id} (${qual.book})` : `Test Verse #${v.index + 1}`;
        return `<option value="${v.index}">${label} — ${escapeHtml(v.ref.slice(0, 48))}...</option>`;
      })
      .join("");
    if (filtered.length > 0) renderVerse(filtered[0].index);
    else verseContainer.innerHTML = "<p>No matching verses found.</p>";
  }

  function renderVerse(idx) {
    const v = data.verses[Number(idx)];
    if (!v) return;
    const qual = qualByIndex.get(v.index);
    const header = qual ? `${qual.verse_id} (${qual.book})` : `Test Verse #${v.index + 1}`;
    verseContainer.innerHTML = `
      <div class="transcript-card">
        <div class="transcript-label">Ground Truth — ${header}</div>
        <div>${escapeHtml(v.ref)}</div>
      </div>
      <div class="transcript-card">
        <div class="transcript-label" style="color:var(--rose)">Config A: Zero-Shot Baseline</div>
        <div>${highlightDiff(v.a, v.ref)}</div>
      </div>
      <div class="transcript-card">
        <div class="transcript-label" style="color:var(--accent)">Config B: LoRA (Raw)</div>
        <div>${highlightDiff(v.b, v.ref)}</div>
      </div>
      <div class="transcript-card">
        <div class="transcript-label" style="color:var(--emerald)">Config C: LoRA + SpecAugment</div>
        <div>${highlightDiff(v.c, v.ref)}</div>
      </div>
    `;
  }

  verseSelect.addEventListener("change", (e) => renderVerse(e.target.value));
  verseSearch.addEventListener("input", (e) => populateSelect(e.target.value));
  populateSelect();
})();
