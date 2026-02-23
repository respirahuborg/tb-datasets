# RespiraHub — TB Cough Screening Research

AI-based TB screening system using cough audio + clinical anamnesis, developed by PT Monago Teknologi Nusantara. Currently in Kemenkes RI regulatory sandbox.

**Core finding after 7 trials:** Clinical metadata alone (AUROC 0.832) matches or outperforms multimodal audio+clinical models (AUROC 0.829) on the CODA TB dataset. Audio contribution ≈ 0 on this dataset — but the hypothesis that audio matters in real Indonesian deployment remains untested.

---

## Dataset: CODA TB

**Source:** [nature.com/articles/s41597-024-03972-z](https://www.nature.com/articles/s41597-024-03972-z)

| Property | Value |
|----------|-------|
| Participants | 1,082 (after train-only filter) |
| Audio segments | 2,216 (dual backbone) / 1,550 (single) |
| Countries | 7 (Uganda, South Africa, Vietnam, dll) |
| Population | Symptomatic only — TB+ vs other respiratory disease (no healthy controls) |
| Ground truth | `tb_status` binary, confirmed via Xpert MTB/RIF |
| Audio type | Solicited cough recordings via Hyfe app |
| Clinical features | 25 columns: sex, age, BMI, cough_dur, hemoptysis, weight_loss, fever, night_sweats, HIV, smoking, country, dll |

### Download Data

```bash
# Clone repo and install dependencies
pip install synapseclient tqdm

# Set environment variable
export SYNAPSE_AUTH_TOKEN=your_token_here  # get at synapse.org

# Run download notebook
jupyter notebook get_data_from_script.ipynb
```

Data will be saved to:
- `data/solicited/` — audio files
- `data/metadata/` — CSV files (clinical, additional, solicited meta)

---

## Setup

```bash
# Install dependencies
pip install torch torchaudio transformers huggingface_hub synapseclient scikit-learn pandas matplotlib tqdm

# Copy env file and fill in tokens
cp .env.example .env

# Set tokens
export HF_TOKEN=hf_...           # huggingface.co/settings/tokens
export SYNAPSE_AUTH_TOKEN=...    # synapse.org Personal Access Tokens
```

---

## Trial Results

| Trial | Model | Approach | AUROC | Key Finding |
|-------|-------|----------|-------|-------------|
| 1 | Wav2Vec2 | Audio baseline, 3s segments | 0.729 ± 0.067 | Baseline confirmed |
| 2 | Wav2Vec2 | Per-file padding | 0.693 ± 0.044 | Too much silence, approach failed |
| 3 | Wav2Vec2 | 5 files/segment | 0.725 ± 0.072 | Best audio preprocessing |
| 4 | Wav2Vec2 | Unfreeze top 2 layers | 0.696 ± 0.074 | Overfitting, freeze is better |
| 5 | HeAR | Google Health Acoustic, 2s | 0.719 ± 0.062 | Health-specific ≠ auto better |
| 6 | W2V + HeAR | Dual-backbone ensemble, 1792d | 0.722 ± 0.071 | Redundant features, not complementary |
| **7a** | **Clinical only** | **Anamnesis, no audio** | **0.832 ± 0.029** | **Clinical dominates** |
| **7b** | **Multimodal** | **Audio + clinical** | **0.829 ± 0.036** | **Audio adds ≈ 0** |
| — | DREAM SC1 winner | Audio-only | 0.743 | Beaten by Trial 7 |
| — | DREAM SC2 winner | Audio + clinical | 0.832 | We match rank 2 |
| — | Zambia study | Audio + clinical (includes HC) | 0.921 | Target with Indonesian data |

---

## Key Insights

### Why audio adds nothing on CODA TB
1. **Country confounder** — 7 countries with different TB prevalence. Model partially learns country, not cough.
2. **Clinical features are too strong** — cough duration, hemoptysis, night sweats, HIV are textbook TB predictors on a curated dataset with near-zero missing values.
3. **Symptomatic population only** — all patients already cough. Acoustic difference TB vs non-TB is too subtle above strong clinical signal.

### Why audio might still matter in Indonesia
1. **Single country** — country confounder eliminated.
2. **Real-world data quality** — anamnesis at puskesmas is often incomplete and inconsistent. Audio = objective signal that cannot be faked.
3. **Healthy controls included** — Indonesian dataset will include HC, making the classification task easier (CODA TB has none).
4. **This is still a hypothesis to be proven.**

---

## Next Research Steps

### Immediate
- [ ] **Deploy clinical scoring tool** — anamnesis-only model (AUROC 0.832) ready to deploy without audio. Immediate clinical value.
- [ ] **Pre-print paper** — draft ablation paper from 7 trials, submit to arXiv/medRxiv to establish priority.
- [ ] **Start IRB process** — biggest bottleneck. Begin now, parallel with other steps.

### Data Collection (Indonesia)
- [ ] **Develop recording app** — anamnesis input + audio recording + Synapse sync.
- [ ] **Phase 1 enrollment** — 180 patients across Jakarta, Surabaya, Yogyakarta (3–4 months post-IRB).
- [ ] **Pilot 20 patients** in Jakarta once IRB approved.

### Modeling
- [ ] **Trial 8: Indonesian data** — HeAR domain adaptation on Indonesian dataset after data collection.
- [ ] **Trial 8/9: Longitudinal coughs** — explore passive cough data (733K+ recordings, untouched). Compare vs solicited. Potential angle: solicited vs passive diagnostic value.
- [ ] **Phase 2 scale** — 365 patients, full paper for journal submission.

---

## Indonesian Data Collection Plan

**Strategy:** Piggyback Xpert — patients undergoing Xpert at faskes → record cough + anamnesis while waiting → label from Xpert result. RespiraHub team does NOT perform diagnosis.

### Phase 1 — 180 patients (3–4 months post-IRB)

| Kota | TB+ | OR | HC | Total |
|------|-----|----|----|-------|
| Jakarta | 25 | 20 | 20 | 65 |
| Surabaya | 25 | 20 | 20 | 65 |
| Yogyakarta | 20 | 15 | 15 | 50 |
| **Total** | **70** | **55** | **55** | **180** |

### Phase 2 — 365 patients (6–9 months)

| Kota | TB+ | OR | HC | Total |
|------|-----|----|----|-------|
| Jakarta | 60 | 40 | 40 | 140 |
| Surabaya | 55 | 35 | 35 | 125 |
| Yogyakarta | 40 | 30 | 30 | 100 |
| **Total** | **155** | **105** | **105** | **365** |

**Groups:**
- **TB+** — Xpert confirmed positive
- **OR** — Symptomatic, Xpert negative
- **HC** — Healthy controls (puskesmas staff, patient companions, volunteers)

> **HC is mandatory.** Without HC, AUROC ceiling ≈ 0.72 (as seen in CODA TB). With HC, target 0.85–0.92 (based on Zambia study).

### Recording Protocol (derived from trial findings)
- Min **5 cough sessions × 2–3 coughs** = 10–15 cough files per patient (Trial 1: 58% patients had only 1 segment — insufficient for soft voting)
- Add **speech recording**: "aah" vowel + counting 1–10 (+30 sec/patient)
- Clinical CSV must match CODA TB columns exactly (plug directly into Trial 7 pipeline)
- Recording format: **44.1kHz WAV mono**, pipeline downsamples to 16kHz

### Target Faskes
- **Jakarta:** RS Persahabatan, PKM Kecamatan (high Xpert volume)
- **Surabaya:** RS Paru Surabaya, RSUD Dr. Soetomo
- **Yogyakarta:** RS Paru Respira, RSUP Sardjito

---

## Paper Strategy

**Strongest finding:** *"Audio Adds Nothing: Clinical Metadata Alone Matches Multimodal TB Screening on CODA TB"* — systematic ablation across 7 trials, negative result challenging core assumption in TB cough screening field.

**Plan:**
1. **Now:** Pre-print (arXiv / medRxiv) — ablation paper from 7 trials. Establish priority.
2. **After Indonesian data:** Full journal paper — domain adaptation + prove/disprove audio value in single-country real-world setting.

**Target journals:** PLOS Digital Health, Lancet Digital Health, Computer Methods and Programs in Biomedicine, ML4H @ NeurIPS/ICML.

---

## Technical Notes

| Item | Detail |
|------|--------|
| Device | Apple Silicon MPS (non-deterministic, ±1–2% variance between runs) |
| Wav2Vec2 | `facebook/wav2vec2-base` — 3s segments, 768d embeddings |
| HeAR | `google/health-acoustic-representations` — 2s segments, 1024d embeddings |
| Dual embeddings | `dual_embeddings.pt` — (2216, 1792d), labels, pids |
| CV strategy | 10-fold stratified, patient-level split, soft voting aggregation |
| Clinical features | BMI from height/weight, symptom_count composite, cough duration buckets (>14d, >30d), country one-hot, StandardScaler fit on train only |

---

## Repository Structure

```
tb-datasets/
├── data/
│   ├── metadata/          # CSV files (gitignored audio)
│   ├── solicited/         # Audio files — gitignored
│   └── longitudinal/      # Passive cough recordings — gitignored, unexplored
├── hear/                  # HeAR repo clone — gitignored
├── checkpoints_t*/        # Model checkpoints — gitignored
├── respirahub_trial*.ipynb
├── get_data_from_script.ipynb
├── respirahub_master_context.md
├── .env.example
└── .gitignore
```
