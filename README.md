# SecureFedHE

**A Decentralized Hybrid Homomorphic Encryption and Differential Privacy Framework for Scalable Federated Learning**

SecureFedHE is a fully decentralized, privacy-preserving federated learning framework designed for hospital-style network environments. It combines a peer-to-peer ring topology with Selective Homomorphic Encryption (CKKS) on the final classification layer and Differential Privacy on intermediate layers, achieving strong privacy guarantees with negligible accuracy loss (~16.5ms encryption overhead per round).

The full research report — architecture, threat model, benchmarking, and security analysis — is available in [`docs/SecureFedHE_MEGA_FINAL.docx`](docs/SecureFedHE_MEGA_FINAL.docx).

---

## Key Results

| Metric | Result |
| --- | --- |
| Accuracy at ε = 10 (primary operating point) | **79.74%** (+0.31% vs unprotected baseline) |
| HE encryption overhead per round | **~16.5ms** (O(1), independent of client count) |
| Round completion under node failure (Fix 1) | **100%** (vs 0% in vanilla ring) |
| Byzantine defence — ZKP commitment scheme (Fix 2) | Poisoned gradients rejected before ring entry |
| Communication payload reduction (Fix 4) | **-25%** size, **-45.6%** round-trip latency |
| Membership Inference Attack accuracy | **52.88%** (near-random) |
| Gradient Inversion (DLG) reconstruction PSNR | **5.18-6.29 dB** (unrecoverable) |

---

## Architecture: The Three Rings

- **Ring 1 — Baseline FedAvg.** Vanilla federated learning, no privacy mechanisms. Reference accuracy: 79.43%. Run locally (`ring1_local/`).
- **Ring 2 — Selective HE + DP.** CKKS homomorphic encryption on the final classifier layer (`fc2`), Differential Privacy noise on the feature-extraction layer (`fc1`) at ε = 10. Run on Google Colab (`ring2_colab/`).
- **Ring 3 — Decentralized Ring Topology.** Serverless peer-to-peer ring where encrypted updates are passed and homomorphically accumulated node-to-node. Run on Google Colab (`ring3_colab/`), with a Flask-based local simulation in `distributed_simulation/`.

---

## Repository Structure

```
SecureFedHE/
├── ring1_local/                  Ring 1 — vanilla FedAvg baseline (run locally)
│   ├── train.py
│   ├── client.py
│   └── aggregator.py
│
├── ring2_colab/                  Ring 2 — Selective HE (CKKS) + DP (run on Colab)
│   ├── ring2.ipynb
│   ├── he_train.py / he_aggregator.py
│   ├── he_layer.py / selective_client.py / validate_he.py
│   └── results/                  he_eps10/20/50.csv, metrics
│
├── ring3_colab/                  Ring 3 — decentralized ring topology (run on Colab)
│   ├── ring3.ipynb
│   └── results/ring_metrics.csv
│
├── distributed_simulation/       Flask-based local P2P ring simulation
│   ├── launch_distributed.py     Launches 3 Flask nodes with ZKP key exchange
│   ├── distributed_node.py       ZKP commitment scheme + key rotation
│   ├── zkp_commitment.py         Cryptographic commitment engine
│   ├── ring_topology.py
│   └── ring_train.py
│
├── large_scale_evaluation/       Large-scale experimental evaluation (Colab)
│   ├── Scalability & Heterogeneity Data/
│   │   ├── notebooks/
│   │   ├── results/               5/10/20/50-client scaling, α=0.1/0.3/0.5/1.0
│   │   └── figures/
│   ├── Architecture & Latency Files/
│   │   ├── notebooks/SecureFedHE_Arch_Latency.ipynb
│   │   ├── results/               SimpleCNN vs ResNet18, 10/50/100ms latency
│   │   └── figures/
│   └── Security Evaluation/
│       ├── notebooks/SecureFedHE_Security_Eval.ipynb
│       ├── results/               MIA, DLG, model-poisoning results
│       └── figures/
│
├── fixes_colab/                  Four engineering fixes (Section 6 of the report)
│   ├── Fix 1 — Ring Topology Resilience/
│   │   ├── notebooks/Fix1_RingFragility_SecureFedHE.ipynb
│   │   ├── results/                fix1_scenario_A/B/C.csv
│   │   └── figures/Figure10_RingFragility_Fix1.png
│   ├── Fix 2 — Byzantine Fault Tolerance/
│   │   ├── notebooks/Fix2_BlindPoisoning_SecureFedHE.ipynb
│   │   ├── results/                25 CSVs: 3 attacks × 4 Byzantine fractions × 2 aggregators
│   │   └── figures/Figure11_BlindPoisoning_Fix2.png
│   └── Fix 3+4 — Positional Bias and Communication Efficiency/
│       ├── notebooks/Fix3_Fix4_DriftBase64_SecureFedHE.ipynb
│       ├── results/                fix3_*.csv (order experiments), fix4_serialisation_benchmark.csv
│       └── figures/                Figure12_WeightDrift_Fix3.png, Figure13_Base64Bloat_Fix4.png
│
├── cross_ring_comparison/         Ring 1 vs 2 vs 3 summary plots & scripts
│   ├── scripts/metrics.py / plot_metrics.py
│   └── plots/                     accuracy_comparison, time_comparison, encryption_overhead
│
├── baseline/                      Shared baseline modules (aggregator, client, training)
├── crypto/                        HE layer and selective encryption modules
├── data/                          Dataset loader (CIFAR-10, non-IID Dirichlet partitioning)
├── network/                       Ring topology and training orchestration
├── models/
│   └── cnn.py                     SimpleCNN architecture (shared across all rings)
│
├── website/                       Public-facing presentation site
├── dashboard/                     CSV-ingestion results dashboard
│
├── docs/
│   └── SecureFedHE_MEGA_FINAL.docx   Full research report
│
├── papers.json                    13-paper comparative literature matrix
└── requirements.txt
```

---

## The Four Engineering Fixes

All four fixes were empirically validated on Google Colab (T4 GPU, CIFAR-10, non-IID Dirichlet partitioning) and are detailed in Section 6 of the report.

### Fix 1 — Ring Topology Resilience (Timeout/Skip Protocol)

**Problem:** In the vanilla Ring 3 topology, any single unresponsive node causes the entire training round to fail (0% completion).

**Fix:** A timeout/skip protocol bypasses non-responsive nodes and re-syncs them in the following round.

**Result:** 100% round completion across all tested failure scenarios (0%, 10% transient, 30% sustained node failure), vs 0% for vanilla Ring 3.

---

### Fix 2 — Byzantine Fault Tolerance (ZKP Commitment Scheme)

**Problem:** The HE aggregator is "blind" — it cannot inspect encrypted submissions, so a malicious node can inject poisoned gradients undetected. An earlier implementation used FedMedian post-decryption, which required a central server and violated the decentralisation guarantee.

**Fix:** A ZKP-inspired cryptographic commitment scheme implemented in `distributed_simulation/zkp_commitment.py`. Before encrypting its fc2 update, each node generates a three-part proof:

- A SHA-256 hash commitment of the plaintext gradient
- The L2 norm value — proves the gradient is within the agreed clipping threshold (C = 0.5)
- An RSA digital signature over both — proves this specific node sent it, unforgeable

The receiving node verifies all three before accepting any ciphertext into the ring. If verification fails, the payload is skipped using the Fix 1 skip protocol. No server is needed and nothing is ever decrypted in transit.

Additional hardening implemented:

- **Replay attack prevention** — round number and timestamp are baked into every signature, so commitments from previous rounds are mathematically rejected
- **Key rotation** — every 2 rounds, each node generates a fresh RSA key pair and broadcasts the new public key to all peers, invalidating any stolen private key

**Architecture reference:** ByzSFL (Fan et al., 2025, arXiv:2501.06953). Full zk-SNARKs remain future work.

**Result:** Byzantine nodes are rejected at the ring boundary before their encrypted payload enters the homomorphic accumulator. No server required. Nothing decrypted.

---

### Fix 3 — Positional Bias (Randomised Node Order)

**Problem:** Fixed traversal order (0→1→2→3→4) in the ring could create positional dominance under non-IID data.

**Fix:** Per-round seeded random permutation of node order — a single additional `numpy` call.

**Result:** Negligible accuracy impact (-0.22pp to +0.36pp); included by default as a fairness guarantee at zero computational cost.

---

### Fix 4 — Communication Efficiency (Raw Bytes Serialisation)

**Problem:** The original Flask transport layer encoded encrypted payloads as Base64 JSON, adding ~33% size overhead.

**Fix:** Transmit raw bytes directly via TenSEAL's `.serialize()`.

**Result:** 25% smaller payload, 45.6% faster round-trip latency — saving ~17.4MB over a 20-round experiment.

---

## Setup

```bash
git clone https://github.com/Varunstyles/SecureFedHE.git
cd SecureFedHE
pip install -r requirements.txt
pip install cryptography
```

**Note:** `tenseal` (used for CKKS homomorphic encryption) requires Linux or macOS. All HE-related experiments (Ring 2, Ring 3, and Fixes 1-4) were run on Google Colab.

### Running Ring 1 (local baseline)

```bash
python ring1_local/train.py
```

### Running Ring 2 / Ring 3 / Fixes (Colab)

Open the relevant `.ipynb` notebook in Google Colab, mount Google Drive, and run cells sequentially. Each notebook is self-contained and documents its own configuration.

### Running the distributed simulation (local)

```bash
python distributed_simulation/launch_distributed.py
```

Launches 3 Flask nodes simulating the Ring 3 peer-to-peer topology on `localhost:5000-5002`. Includes automatic ZKP public key exchange before training and key rotation every 2 rounds.

---

## Citation

If you use this work, please cite the accompanying report in `docs/SecureFedHE_MEGA_FINAL.docx`.

## License

[Specify your license here]
