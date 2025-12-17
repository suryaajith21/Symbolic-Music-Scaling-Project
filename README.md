# Symbolic Music Scaling Project (Lakh MIDI → ABC)

This repository contains my CS-GY 6923 project: **Scaling Laws for Symbolic Music Modeling**.

I study whether **neural scaling laws** (Kaplan-style power laws) appear in **symbolic music** by training:
- a family of **GPT-style decoder-only Transformers** (Tiny → XL) and
- **LSTM** baselines (Tiny → XL)
on **Lakh MIDI converted to ABC notation**.

I also train a stronger Phase-2 **BEST** model (BPE tokenizer + improved preprocessing) focused on generation quality.

---

## Key results

### Scaling behavior
- Symbolic music modeling shows power-law scaling with an exponent **α ≈ 0.08**.
- **LSTMs** are more sample-efficient below ~**5M params**, but **Transformers** scale better beyond that.

### BEST model evaluation
- **Test perplexity:** 1.9336  
- **Syntactic validity:** 68.0% (music21 parses generated ABC)  
- **ABC → MIDI success:** 58.8% (music21 MIDI export succeeds)

---

## Repository structure

```text
Symbolic-Music-Scaling-Project/
src/ # Phase 1: scaling study scripts (char-level)
Best Model/
src/ # Phase 2: BEST model preprocessing + tokenizer + generation scripts
notebooks/ # Colab notebooks (training + evaluation)
unconditional_generation_samples/ # Example unconditional generations (.abc)
conditional generation samples/ # Example prompted/conditional generations (.abc)
requirements.txt
README.md
```

### Folder structure

### Folder structure

### Phase 1 — Scaling Study (`src/`)
Typical contents:
- `process_data.py` : Phase-1 preprocessing / dataset construction
- `model.py`, `config.py` : model/config helpers
- `plot.py`, `plot_train.py` : plotting scaling + training curves
- `conversion_stats.py`, `count_tokens.py`, `report_data.py` : stats + reporting helpers

### Phase 2 — BEST Model (`Best Model/src/`)
Typical contents:
- `gzip_stats.py` : gzip-ratio diagnostics
- `process_safe.py` : improved cleaning + filtering pipeline
- `split_songs.py` : song splitting utilities
- `train_tokenizer.py` : train BPE tokenizer (vocab 4096)
- `tokenize_to_bin.py` : tokenize dataset into training-ready format
- `generate_unconditional.py`, `generate_local.py` : generation scripts
- `count_tokens.py` : token counting for Phase 2 corpora

### Colab notebooks (`notebooks/`)
- `Music_Scaling.ipynb` : scaling study runs
- `TRAIN_BEST_MODEL.ipynb` : BEST model training
- `Perplexity.ipynb` : evaluation notebook(s)
- `01_data.ipynb` : File download

---

## Setup

Create an environment and install dependencies:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Data

This repo does not include the raw Lakh MIDI dataset, converted ABC corpora, or large training artifacts.
You must download Lakh MIDI separately and convert MIDI → ABC (e.g., using midi2abc or music21) like shown in 'notebooks\01_data.ipynb'.

---

## Phase 1: Scaling Study (character-level)

Goal: fit a scaling curve by training multiple model sizes under a fixed compute/token budget.

### Training budget (Phase 1)
- MAX_TRAIN_TOKENS = 100,000,000
- BATCH_SIZE = 64, BLOCK_SIZE = 256
- Tokens/step = 64 × 256 = 16,384
- Total steps = ⌊100,000,000 / 16,384⌋ = 6,103 steps

## How to run (Phase 1)
- Convert MIDI → ABC and point your preprocessing script at your local ABC directory.
- Run Phase-1 preprocessing to build train/val/test text files.
- Train Tiny → XL models (training was run on Colab in this project; see notebooks/).
- Use src/plot.py / src/plot_train.py to reproduce scaling plots and training curves.
- Note: GPT-style training is based on nanoGPT with modifications for ABC preprocessing/tokenization.

---

## Phase 2: BEST Model (BPE + improved preprocessing)

Goal: improve generation quality by fixing Phase-1 failure modes and using a BPE tokenizer.

### Phase-2 pipeline summary
- Gzip-ratio quality filtering: keep file if gzip_bytes / raw_bytes >= 0.24
- Leading-silence trimming: trims rest-heavy leading lines to reduce silence loops
- Explicit song delimiter: each song ends with \n<|endoftext|>\n
- BPE tokenizer: vocab size 4096
- BEST model: ~206M parameters, context window 1024, trained 11,000 steps

## How to run (Phase 2)
```bash
python "Best Model/src/gzip_stats.py"
python "Best Model/src/process_safe.py"
python "Best Model/src/train_tokenizer.py"
python "Best Model/src/tokenize_to_bin.py"
```

Then train on Colab:
notebooks/TRAIN_BEST_MODEL.ipynb

---

## Generating samples

### Unconditional generation
Script: Best Model/src/generate_unconditional.py

Example outputs: unconditional_generation_samples/

## Conditional / prompted generation
Script: Best Model/src/generate_local.py

Example outputs: conditional generation samples/

---

## Model configurations (report)

### Transformers (GPT-style):
- Tiny: 0.8M (4 layers, d=128, 4 heads, vocab 64)
- Small: 4.8M (6 layers, d=256, 8 heads, vocab 64)
- Medium: 21.4M (12 layers, d=384, 6 heads, vocab 64)
- Large: 50.5M (16 layers, d=512, 8 heads, vocab 64)
- XL: 113.5M (16 layers, d=768, 12 heads, vocab 64)
- BEST: 206M (16 layers, d=1024, 16 heads, vocab 4096)

### LSTM baselines:
- Tiny: ~1.07M
- Small: ~5.06M
- Medium: ~19.45M
- Large: ~49.14M
- XL: ~100.20M

---

## References
- Kaplan et al. (2020) — Scaling Laws for Neural Language Models
- Radford et al. (2019) — GPT-2
- Karpathy (2022) — nanoGPT
- Raffel (2016) — Lakh MIDI Dataset (LMD)
