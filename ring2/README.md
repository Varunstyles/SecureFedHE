# Ring 2 — Selective HE Implementation

## What's in this folder

These files replace/add to your existing `securefedhe` project:

```
ring2/
├── crypto/
│   ├── he_layer.py          ← REPLACE the stub (full CKKS + DP implementation)
│   ├── selective_client.py  ← NEW (SelectiveHEClient class)
│   └── validate_he.py       ← NEW (sanity-check script, run first)
└── baseline/
    ├── he_aggregator.py     ← NEW (server-side HE aggregation)
    └── he_train.py          ← NEW (Ring 2 training entry point)
```

---

## Setup (Linux / WSL / Google Colab)

```bash
# 1. Install TenSEAL (Linux only — does not work on Windows)
pip install tenseal

# 2. Copy files into your existing securefedhe project
cp ring2/crypto/he_layer.py        securefedhe/crypto/he_layer.py
cp ring2/crypto/selective_client.py securefedhe/crypto/selective_client.py
cp ring2/crypto/validate_he.py      securefedhe/crypto/validate_he.py
cp ring2/baseline/he_aggregator.py  securefedhe/baseline/he_aggregator.py
cp ring2/baseline/he_train.py       securefedhe/baseline/he_train.py
```

---

## Running Ring 2

**Step 1 — Always validate HE first:**
```bash
cd securefedhe
python -m crypto.validate_he
```

Expected output:
```
[1] TenSEAL import ........... OK  (version 0.3.x)
[2] Context creation ......... OK  (1.2s)
[3] Encrypt → Decrypt ........ OK
[4] Approximation error ...... OK  (max=3.2e-06)
[5] HE aggregation ........... OK  (max_diff=4.1e-06)
[6] Encryption time .......... OK  (0.034s per round)
✓  All checks passed.
```

**Step 2 — Quick smoke test (5 rounds):**
```bash
python -m baseline.he_train --rounds 5 --clients 3
```

**Step 3 — Full benchmark run (matches Ring 1 settings):**
```bash
python -m baseline.he_train --rounds 20 --clients 5 --epsilon 2.0
```

---

## Ring 2 Milestone Gate

Before moving to Ring 3, confirm ALL three:

| Check | Target | Where to look |
|---|---|---|
| Accuracy drop vs Ring 1 | < 0.5% | Terminal output (79.43% → ≥ 78.93%) |
| HE approx error | < 1e-3 | validate_he.py output |
| enc_overhead_s | > 0 (should be ~0.03–0.1s) | he_metrics.csv |

---

## Key numbers for your paper (Table 2)

From `evaluation/logs/he_metrics.csv`, report:

- **Average `enc_overhead_s`** — the cost of selective encryption per round
- **Average `wall_time_s`** vs baseline → compute overhead ratio
- **Average `comm_bytes`** vs baseline → communication cost ratio  
- **Final `eval_acc`** vs baseline → accuracy preservation

The thesis of your paper: selective HE achieves comparable accuracy
to full HE with 3-5x less overhead, and comparable privacy guarantees.

---

## DP Privacy Parameters Guide

| ε (epsilon) | Privacy Level | Expected Accuracy Impact |
|---|---|---|
| 10.0 | Weak | < 0.1% drop |
| 2.0  | Moderate (recommended) | ~0.2–0.5% drop |
| 1.0  | Strong | ~1–2% drop |
| 0.1  | Very strong | ~3–5% drop |

Start with ε=2.0. Run experiments at ε=1.0 and ε=10.0 for your
privacy-utility trade-off curve (this becomes a figure in your paper).
