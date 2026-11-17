# Phase 4 Evaluation Report: Yoruba BibleTTS ASR

## Task 4.1: Diacritic-Preserving vs. Diacritic-Stripped Breakdown

| Configuration | WER (%) | WER No-Tone (%) | Diacritic WER Gap | CER (%) | CER No-Tone (%) | Diacritic CER Gap |
|---|---:|---:|---:|---:|---:|---:|
| Config A: Zero-Shot Baseline | 113.45 | 109.36 | +4.09 | 81.63 | 71.34 | +10.29 |
| Config B: LoRA (Raw) | 48.51 | 39.49 | +9.02 | 18.02 | 12.98 | +5.05 |
| Config C: LoRA + SpecAugment | 49.96 | 40.23 | +9.73 | 17.44 | 11.60 | +5.84 |

## Task 4.2: Theological Named Entity Error Audit (T-NEER)

| Configuration | Precision (%) | Recall (%) | F1 (%) | True Positives | False Negatives |
|---|---:|---:|---:|---:|---:|
| Config A: Zero-Shot Baseline | 0.00 | 0.00 | 0.00 | 0 | 104 |
| Config B: LoRA (Raw) | 100.00 | 71.15 | 83.15 | 74 | 30 |
| Config C: LoRA + SpecAugment | 98.63 | 69.23 | 81.36 | 72 | 32 |

## Task 4.3: Qualitative Side-by-Side Verse Comparisons

### EPH_001_Verse_013 (Book: `EPH`)
- **Theological Entities:** Ọlọ́run, Olúwa, Ẹ̀mí
- **Ground Truth:** Ògo ni fún Ọlọ́run àti Baba Jesu Kristi Olúwa wa, ẹni tí ó ti bùkún wa láti inú ọ̀run wá pẹ̀lú àwọn ìbùkún ẹ̀mí gbogbo nínú Kristi.
- **Config A: Zero-Shot Baseline:**  ḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍ
- **Config B: LoRA (Raw):** Ìgó ní fún ọlọ́run àti baba Jesu Kristi Olúwa wá, ẹni tó ti bòkún wá láti nú ọ̀run wa pẹ̀lú àwọn ìbòkún ẹmí gbogbo nínú Kristi.
- **Config C: LoRA + SpecAugment:** Ìgó ní fún ọlọ́run àti baba Jesu Kristi Olúwa wá, ẹni tó ti bòkún wá láti nú ọ̀run wa pẹ̀lú àwọn ìbòkún ẹmí gbogbo níno Kristi.

### GAL_004_Verse_019 (Book: `GAL`)
- **Theological Entities:** Ọlọ́run, ẹ̀ṣẹ̀, Mímọ́
- **Ground Truth:** Ṣùgbọ́n nísinsin yìí, nígbà tí ẹ̀yin ti mọ Ọlọ́run tan tàbí kí a sá kúkú wí pé ẹ di mímọ́ fún Ọlọ́run, èéha ti rí tí ẹ tún fi yípadà sí aláìlera àti agbára ìpilẹ̀ṣẹ̀ ẹ̀dá, lábẹ́ èyí tí ẹ̀yin tún fẹ́ padà wá ṣe ẹrú?
- **Config A: Zero-Shot Baseline:**  Shubonis yi sin yi, nibati yi yi ti molon uton tabiki yi asha kukubi kwe yi edimimofung olonon, yi atiri ti etun fi yi pada si yi alailira ati agbara ek pile se yi da, labeyi ti yi tun fe kwada washieru.
- **Config B: LoRA (Raw):** Ṣùgbọ́n nísìsìn yìí, nígbà tí ẹ̀yin ti mọ́ Ọlọ́run tán tàbí kí a ṣá àkùkù wí pé ẹ̀dí mímọ́ fún Ọlọ́run, èyí a ti rí tí ẹ tún fi yí padà sí ìà láìlìrà àti agbára ìpílẹ̀sẹ̀ ẹ̀dá, lábẹ̀ yìí tí ẹ̀yin tún fẹ́ padà wá ṣe rú.
- **Config C: LoRA + SpecAugment:** Ṣùgbọ́n ní ṣin yìí, nígbà tí ẹ̀yin ti mọ́ ọlọ́run tán tàbí kí a ṣá àkùkù wí pé ẹ̀dí mímọ́ fún ọlọ́run, èyí a ti rí tí ẹ̀tún fi yí ìpàdà sí ìà láìlìrà àti agbára ìpílẹ̀sẹ̀ ẹ̀dá, lábẹ̀ yìí tí ẹ̀yin tún fẹ́pàdà wá ṣe ẹ̀rú.

### COL_001_Verse_014 (Book: `COL`)
- **Theological Entities:** Ọlọ́run, oore-ọ̀fẹ́, Mímọ́
- **Ground Truth:** Sí àwọn ẹni mímọ́ àti àwọn arákùnrin àti arábìnrin nínú Kristi tí wọ́n ń gbé ní ìlú Kolose: Oore-ọ̀fẹ́ àti àlàáfíà fún un yín láti ọ̀dọ̀ Ọlọ́run Baba wa.
- **Config A: Zero-Shot Baseline:**  Sī yāwenimimo āt'yā warakuni āt'yāra breni nukristi āt'yārng breni lukulose āt'yārng breni lukulose Uriofe āt'yārng fiafungi āt'yārng dōlōn o bābāwa āt'yārng dōlōn o bābāwa āt'yārng dōlōn o bābāwa āt'yārng dōlōn o bābāwa āt'yārng dōlōn o bābāwa āt'yārng dōlōn o bābāwa āt'yārng dōlōn o bābāwa āt'yārng dōlōn o b
- **Config B: LoRA (Raw):** Sí àwọn èní mímọ́ àti àwọn arákùnrin àti arabìnrin ní nú Kristi tí wọn ń gbé ní ìlúkólóse. Orí ọ̀fẹ́ àti àláfíà fún yín láti ọ̀dọ̀ Ọlọ́run baba wà.
- **Config C: LoRA + SpecAugment:** Ṣí àwọn ẹni mímọ́ àti àwọn arákùnrin àti arabìnrin ní Nú Kristi tí wọn gbé ní ìlúkólóse. Orí ọ̀fẹ́ àti àláfíà fún yín láti ọ̀dọ̀ Ọlọ́n-òmpapá wa.

### EPH_001_Verse_002 (Book: `EPH`)
- **Theological Entities:** ìgbàlà, Ẹ̀mí, Mímọ́
- **Ground Truth:** Àti ẹ̀yin pẹ̀lú darapọ̀ nínú Kristi nígbà tí ẹ̀yin gbọ́ ọ̀rọ̀ òtítọ́ náà àní ìyìnrere ìgbàlà yin. Nígbà tí ẹ̀yin gbàgbọ́, a fi Ẹ̀mí Mímọ́ ṣe èdìdì ayé yin nínú rẹ, èyí tí a ti ṣe ìlérí rẹ̀ tẹ́lẹ̀,
- **Config A: Zero-Shot Baseline:**  ḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍḍ
- **Config B: LoRA (Raw):** Àti ẹ̀yin pẹ̀lú òdárapọ̀ nínú Kristi nígbà tí ẹ̀yin gbọ́ ọ̀rọ̀ títọ́ náà, àní ìrírí re ìgbà láyín, nígbà tí ẹ̀yin gbà gbọ́, a fi ẹ̀mí mí wọ́n ṣe ìdìdìyá yí yín nínú rẹ̀, èyí tí a ti ṣe ilérí rẹ̀ tẹ́lẹ̀.
- **Config C: LoRA + SpecAugment:** Àti èyín pẹ̀lú òdárapọ̀ nínú Kristi nígbà tí ẹ̀yín gbọ́ ọ̀rọ̀ títọ́ náà, àní inrírí re ìgbà láyín, nígbà tí ẹ̀yín gbà gbọ́, a fi ẹ̀mí mímọ́ ṣe ìdìdìyà yí yín nínú rẹ̀, èyí tí a ti ṣe ilé rí rẹ̀ tẹ́lẹ̀.

### EPH_004_Verse_003 (Book: `EPH`)
- **Theological Entities:** Ọlọ́run, òdodo, Mímọ́
- **Ground Truth:** kí ẹ sì gbé ọkùnrin tuntun ni wọ̀, èyí tí a dá ni àwòrán Ọlọ́run nínú òdodo àti ìwà mímọ́ ti òtítọ́.
- **Config A: Zero-Shot Baseline:**  kī esi bē yokone t'u tūni wo e iti adani yawonolonu inu dodu a tī wā mimo t'u titcho.
- **Config B: LoRA (Raw):** Kí ẹ̀sìgbé ọkùnrin tún tún ní wọ̀, èyí ti a dá ní àwọn ọ̀rọ̀nrọ̀nrọ̀nrún nínú dòdò àti ìwàmímọ́ ti òtítọ́.
- **Config C: LoRA + SpecAugment:** Kí ẹ̀ṣìgbé ọkùnrin tún tún ní wọ̀, èyí ti a dá ní àwọn ọ̀rọ̀lọ́run nínú dodó àti ìwà mímọ́ ti òtítọ́.
