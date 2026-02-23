# RespiraHub — Master Context Document
## Untuk Konteks Sesi Chat Baru
**Last updated: 22 Februari 2026**

---

## 1. Apa itu RespiraHub

RespiraHub adalah sistem skrining TB berbasis AI yang menganalisis suara batuk + data anamnesis melalui smartphone. Dikembangkan oleh Husein (PT Monago Teknologi Nusantara). Saat ini dalam regulatory sandbox Kemenkes RI.

**Posisi produk (updated setelah Trial 7):** Digital triage tool yang menstandarisasi penilaian risiko TB — bukan hanya "AI yang dengerin batuk." Audio batuk sebagai sinyal objektif tambahan, clinical scoring sebagai backbone utama.

---

## 2. Dataset: CODA TB

- **Source:** nature.com/articles/s41597-024-03972-z
- **Size:** 1.082 pasien (setelah filter train-only), 2.216 segments (dual backbone), 1.550 segments (single backbone)
- **Countries:** 7 negara (Uganda, South Africa, Vietnam, dll)
- **Population:** Semua symptomatic (TB+ vs respiratory disease lain). TIDAK ada healthy controls.
- **Labels:** tb_status binary (0/1), confirmed via Xpert MTB/RIF
- **Audio:** Solicited cough recordings, Hyfe app, variable duration per patient
- **Clinical metadata:** 25 columns — sex, age, height, weight, cough_dur, hemoptysis, weight_loss, fever, night_sweats, tb_prior, HIV, smoking, heart_rate, temperature, country, dll. Almost zero missing values.
- **Paths di local machine:**
  - Audio: `/Users/aida/code/development/tb-datasets/data/solicited/`
  - Clinical: `/Users/aida/code/development/tb-datasets/data/metadata/CODA_TB_Clinical_Meta_Info.csv`
  - Additional: `/Users/aida/code/development/tb-datasets/data/metadata/CODA_TB_additional_variables_train.csv`
  - Solicited: `/Users/aida/code/development/tb-datasets/data/metadata/CODA_TB_Solicited_Meta_Info.csv`

---

## 3. Semua Trial Results

### Trial 1: Wav2Vec2 Baseline
- **Config:** Wav2Vec2-base, feature_extractor frozen, transformer trainable. 3s segments (concat all coughs + split). Batch 4, grad accum 8 (effective 32), LR 3e-5, 5 epochs.
- **Result:** AUROC **0.729 ± 0.067** (re-run v2). Original run: 0.733.
- **Per-fold:** [0.691, 0.738, 0.826, 0.860, 0.675, 0.769, 0.687, 0.646, 0.676, 0.723]
- **Finding:** 58% pasien hanya punya 1 segment. Fold 4 consistently best, Fold 8 worst.
- **Files:** `respirahub_trial1_v2.ipynb`, `respirahub_trial1_report.pdf` (report lama, AUROC 0.733)

### Trial 2: Per-File Padding
- **Config:** Setiap file batuk individual di-pad ke 3s (bukan concat semua dulu). Tujuan: lebih banyak segments per pasien.
- **Result:** AUROC **0.693 ± 0.044**
- **Finding:** Terlalu banyak silence (file batuk cuma 0.5s, padding 2.5s silence). Model belajar silence, bukan batuk. Approach gagal.

### Trial 3: 5 Files per Segment
- **Config:** Gabungkan 5 file batuk per segment (bukan semua sekaligus). Sweet spot antara konten dan padding.
- **Result:** AUROC **0.725 ± 0.072**
- **Finding:** Best audio preprocessing approach. Tapi improvement marginal vs Trial 1.

### Trial 4: Unfreeze 2 Transformer Layers
- **Config:** Sama dengan Trial 1, tapi unfreeze top 2 transformer layers Wav2Vec2.
- **Result:** AUROC **0.696 ± 0.074**
- **Finding:** Overfitting. Dataset terlalu kecil untuk fine-tune transformer layers. Freeze is better.

### Trial 5: HeAR (Google Health Acoustic)
- **Config:** google/health-acoustic-representations, 2s segments. Frozen embeddings (1024d) + MLP head.
- **Result:** AUROC **0.719 ± 0.062**
- **Finding:** Health-specific pre-training ≠ automatically better. Setara Wav2Vec2. Tapi 10x lebih cepat karena frozen embeddings.
- **Files:** `respirahub_trial5_report.docx`

### Trial 6: Dual-Backbone Ensemble
- **Config:** Wav2Vec2 (768d) + HeAR (1024d) = 1792d concatenated. MLP 3-layer (1792→256→64→1), ~475K params.
- **Result:** AUROC **0.722 ± 0.071**
- **Finding:** Marginal improvement (+0.003 vs HeAR). Kedua model capture sinyal yang sama (redundant, not complementary). Fold patterns identical across all trials = bottleneck di data, bukan model.
- **Files:** `respirahub_trial6.ipynb`, `respirahub_trial6_report.docx`, `dual_embeddings.pt` (2216 × 1792d)

### Trial 7: Multimodal (Audio + Clinical) ⭐
- **Config:** Two-branch architecture. Audio branch: 1792d → 256 → 128. Clinical branch: 25d → 64 → 64. Fusion: 192 → 64 → 1. Total 514K params.
- **Clinical features (25):** 6 continuous (age, BMI, heart_rate, temperature, cough_dur, symptom_count) + 12 binary (sex, HIV, hemoptysis, weight_loss, fever, night_sweats, smoking, tb_prior variants, cough >14d, >30d) + 7 country one-hot
- **Result:** AUROC **0.829 ± 0.036**
- **Per-fold:** [0.828, 0.860, 0.842, 0.887, 0.805, 0.846, 0.865, 0.776, 0.772, 0.811]
- **CRITICAL FINDING:** Clinical-only baseline = **0.832 ± 0.029**. Audio adds **-0.003** (literally zero/negative). Seluruh predictive power dari clinical features.
- **DREAM Challenge context:** Subchallenge 2 (audio+clinical) winner Metformin-121 = 0.832. Kita 0.829 = setara rank 2.
- **Threshold table (multimodal):**
  - 0.10: Sens 89.3%, Spec 54.6% (almost WHO sens target)
  - 0.25: Sens 78.7%, Spec 71.3% (spec passes WHO 70%)
  - 0.40: Sens 70.1%, Spec 80.2% (Youden optimal)
  - WHO TPP (sens ≥90% + spec ≥70%) NOT MET simultaneously
- **Files:** `respirahub_trial7.ipynb`, `respirahub_trial7_report.docx`

---

## 4. Summary Table

| Trial | Model | Approach | AUROC | Key Finding |
|-------|-------|----------|-------|-------------|
| 1 | Wav2Vec2 | Audio baseline 3s | 0.729 | Baseline confirmed |
| 2 | Wav2Vec2 | Per-file padding | 0.693 | Too much silence |
| 3 | Wav2Vec2 | 5 files/segment | 0.725 | Best audio preproc |
| 4 | Wav2Vec2 | Unfreeze 2 layers | 0.696 | Overfitting |
| 5 | HeAR | Health acoustic 2s | 0.719 | Health ≠ auto better |
| 6 | W2V+HeAR | Ensemble 1792d | 0.722 | Redundant features |
| 7a | Clinical only | Anamnesis, no audio | **0.832** | Clinical dominates |
| 7b | Multimodal | Audio + clinical | **0.829** | Audio adds ≈0 |
| — | DREAM SC1 winner | Audio-only | 0.743 | Beaten by Trial 7 |
| — | DREAM SC2 winner | Audio+clinical | 0.832 | We match rank 2 |
| — | Zambia study | Audio+clinical | 0.921 | Target (includes HC) |

---

## 5. Key Insights & Decisions

### Why audio adds nothing on CODA TB:
1. **Country confounder** — 7 countries dengan prevalensi TB berbeda. Model "cheat" pakai country feature.
2. **Clinical features terlalu strong** — cough duration, hemoptysis, night sweats, HIV = textbook TB predictors. Data curated, almost zero missing.
3. **Symptomatic population** — semua pasien sudah batuk. Acoustic difference TB vs non-TB terlalu subtle di atas sinyal klinis yang kuat.

### Why audio might still matter in Indonesia:
1. **Single country** — country confounder hilang.
2. **Real-world data quality** — anamnesis di puskesmas sering incomplete, inconsistent. Audio = sinyal objektif yang ga bisa di-fake.
3. **Healthy controls** — dataset Indonesia akan include HC, bikin task lebih mudah (CODA TB ga punya HC).
4. **Ini masih HYPOTHESIS yang perlu dibuktikan.**

### Product positioning:
- BUKAN "AI lebih pintar dari dokter"
- ADALAH "Digital triage tool yang menstandarisasi penilaian risiko TB"
- Consistency (same output regardless of operator), Access (70% puskesmas tanpa Xpert), Audio sebagai insurance untuk real-world deployment

---

## 6. Documents Generated

| File | Description |
|------|-------------|
| `respirahub_trial1_v2.ipynb` | Trial 1 notebook (re-run, AUROC 0.729) |
| `respirahub_trial5_report.docx` | Trial 5 HeAR report (Bahasa Indonesia) |
| `respirahub_trial6.ipynb` | Trial 6 dual-backbone ensemble notebook |
| `respirahub_trial6_report.docx` | Trial 6 report dengan analisis why ensemble failed |
| `respirahub_trial7.ipynb` | Trial 7 multimodal notebook (audio + clinical) |
| `respirahub_trial7_report.docx` | Trial 7 report — MOST IMPORTANT, includes honest analysis audio=0 |
| `respirahub_sop_skrining.docx` | SOP Skrining TB di Puskesmas/RS, 10 bab lengkap |
| `respirahub_tasklist_v2.docx` | Task list data collection v2.0, 3 kota, 5 fase |
| `dual_embeddings.pt` | Pre-extracted Wav2Vec2+HeAR embeddings (2216 × 1792d) |

---

## 7. Data Collection Plan (Indonesia)

### Strategy: Piggyback Xpert
Pasien yang di-Xpert oleh dokter faskes → rekam batuk + anamnesis sambil nunggu hasil → label otomatis dari Xpert result. Tim RespiraHub TIDAK melakukan diagnosis sendiri.

### 3 Kota, 2 Phase:

**Phase 1 (180 pasien, 3-4 bulan):**
| Kota | TB+ | OR | HC | Total |
|------|-----|----|----|-------|
| Jakarta | 25 | 20 | 20 | 65 |
| Surabaya | 25 | 20 | 20 | 65 |
| Yogyakarta | 20 | 15 | 15 | 50 |
| **Total** | **70** | **55** | **55** | **180** |

**Phase 2 (365 pasien, 6-9 bulan):**
| Kota | TB+ | OR | HC | Total |
|------|-----|----|----|-------|
| Jakarta | 60 | 40 | 40 | 140 |
| Surabaya | 55 | 35 | 35 | 125 |
| Yogyakarta | 40 | 30 | 30 | 100 |
| **Total** | **155** | **105** | **105** | **365** |

### Kelompok:
- **TB+** = Xpert confirmed positive (oleh dokter faskes)
- **OR** = Symptomatic, Xpert negative
- **HC** = Healthy controls (petugas puskesmas, keluarga pengantar, volunteers)
- **HC WAJIB** — tanpa HC, AUROC mentok 0.72 (CODA TB). Dengan HC, target 0.85-0.92 (Zambia).

### Key adjustments dari 7 trial:
- Min 5 sesi batuk × 2-3 batuk = 10-15 cough files per pasien (Trial 1: 58% pasien cuma 1 segment)
- Tambah speech recording: "aah" vowel + counting 1-10 (+30 detik/pasien)
- Clinical CSV harus match CODA TB columns exactly (plug langsung ke Trial 7 pipeline)
- Recording: 44.1kHz WAV mono, pipeline downsample ke 16kHz

### Faskes target:
- Jakarta: RS Persahabatan, PKM Kecamatan (high Xpert volume)
- Surabaya: RS Paru Surabaya, RSUD Dr. Soetomo
- Yogyakarta: RS Paru Respira, RSUP Sardjito

### Timeline:
- Bulan 1-2: IRB approval + MoU faskes (bottleneck)
- Bulan 2-3: Protokol + setup + app deploy
- Bulan 3: Pilot 1 site Jakarta (20 pasien)
- Bulan 3-6: Phase 1 enrollment (180 pasien, 3 kota parallel)
- Bulan 6-7: Data management + handoff
- Bulan 7-8: Trial 8 — HeAR domain adaptation on Indonesian data
- Bulan 6-12: Phase 2 scale (365 pasien) + paper submission

---

## 8. Paper Strategy

### Strongest publishable finding:
"Audio Adds Nothing: Clinical Metadata Alone Matches Multimodal TB Screening on CODA TB" — systematic ablation study, 7 trials, negative result that challenges core assumption in TB cough screening field.

### Strategy: Opsi C (recommended)
1. **Now:** Pre-print (arXiv/medRxiv) — ablation paper based on 7 trials. Establish priority.
2. **After Indonesian data:** Full paper for journal (Lancet Digital Health, PLOS Digital Health) — domain adaptation + prove/disprove audio value in single-country setting.

### Target journals:
- PLOS Digital Health
- Computer Methods and Programs in Biomedicine
- ML4H workshop (NeurIPS/ICML)
- Lancet Digital Health (kalau Indonesian data strong)

---

## 9. Next Steps (Prioritized)

1. **Deploy clinical scoring tool** — anamnesis-only model (AUROC 0.832) bisa deploy tanpa audio. Immediate value.
2. **Start IRB process** — bottleneck terbesar. Mulai sekarang, parallel dengan lainnya.
3. **Pre-print paper** — draft ablation paper dari 7 trial, submit ke arXiv/medRxiv.
4. **Develop recording app** — RespiraHub app: anamnesis input + audio recording + sync.
5. **Pilot data collection** — begitu IRB approved, pilot 20 pasien di Jakarta.
6. **Trial 8** — HeAR domain adaptation on Indonesian data (setelah data ready).

---

## 10. Belum Di-Explore: Longitudinal (Passive) Cough Data

CODA TB dataset punya 2 jenis rekaman batuk:
- **Solicited coughs** — diminta batuk di depan HP. Ini yang kita pakai di semua 7 trial.
- **Longitudinal/passive coughs** — rekaman batuk natural yang dikumpulkan terus-menerus pakai Hyfe app selama berhari-hari/berminggu-minggu. Jumlahnya jauh lebih banyak (733K+ total recordings di CODA TB).

**Kenapa belum di-explore:**
- 7 trial fokus di solicited dulu karena itu yang paling comparable dengan deployment di puskesmas (pasien diminta batuk saat skrining).
- Solicited coughs lebih controlled = lebih mudah di-standardize.

**Kenapa worth exploring:**
- Volume data jauh lebih besar per pasien → soft voting lebih efektif.
- Natural coughs mungkin capture pola batuk yang berbeda dari solicited (forced vs spontaneous).
- Bisa jadi approach untuk passive monitoring (bukan one-time screening tapi continuous).
- TBscreen paper (Science Advances) pakai passive cough classifier, dapet akurasi 82%.

**Potential Trial 8 atau 9:** Train model di longitudinal data dan bandingkan vs solicited. Atau combine keduanya. Ini bisa jadi angle menarik untuk paper juga — solicited vs passive cough diagnostic value.

**Path di dataset:** Perlu cek folder structure CODA TB untuk longitudinal recordings (kemungkinan di folder terpisah dari `/solicited/`). Metadata mungkin di file CSV yang berbeda.

---

## 11. Technical Notes

- **Device:** Apple Silicon MPS (non-deterministic, ±1-2% variance antar run)
- **HeAR model:** `google/health-acoustic-representations` — 2s segments, 1024d embeddings
- **Wav2Vec2:** `facebook/wav2vec2-base` — 3s segments, 768d embeddings
- **Dual embeddings file:** `dual_embeddings.pt` — contains combined_embeddings (2216, 1792), labels, pids
- **Training:** All trials use 10-fold stratified CV, patient-level split, soft voting aggregation
- **Clinical features engineering:** BMI computed from height/weight, symptom_count composite, cough duration buckets (>14d, >30d), country one-hot. Continuous features StandardScaler fit on train only.
