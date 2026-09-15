# Novelty check: "What does the student inherit?"

**Date:** 15 Sep 2026 · **Verdict:** the gap is still open.

No paper was found that distils a traffic model (foundation model or otherwise) into a small student and tests whether the student keeps the teacher's unknown-traffic detection, calibration and robustness over time, using time-based splits on CESNET data.

## How it was checked
- **Web searches:** KD combined with open-set, unknown, OOD, calibration, shortcuts, conformal prediction or drift; English and Chinese.
- **Citing papers (Semantic Scholar / OpenAlex / Crossref):** papers citing CESNET-TLS-Year22, the TIFS 2025 open-set paper, NetClus and ResAware.
  - No paper citing CESNET-TLS-Year22 uses distillation.
  - ResAware's only citers are CipherSight and the PQC "Colossus" paper, both website-fingerprinting work.
- **Limits:**
  - ScienceDirect, Springer and MDPI full texts were blocked.
  - Semantic Scholar rate-limited some lookups.
  - Google Scholar and IEEE Xplore could not be queried; a manual search there is still worth doing.

## Closest papers

| Paper | Overlap | Why it does not cover the idea |
|---|---|---|
| [NetClus](https://arxiv.org/abs/2508.02282) (under review) | Distils TrafficFormer/YaTC into a 5-layer feed-forward network; flags new traffic types | One held-out malware class; random 8:1:1 splits; CSTNET-TLS1.3, ISCX-VPN, USTC-TFC; no calibration, drift or shortcut analysis |
| [ResAware](https://arxiv.org/abs/2606.17462) (June 2026) | Distilled student gains open-world TPR (22.4% → 27.2%) and calibration (ECE 0.138 → 0.034) under 150-day drift | Website fingerprinting only; no application classification, CESNET data, foundation models or conformal prediction |
| [PreDyn-IDS](https://doi.org/10.1016/j.inffus.2026.104611) (Information Fusion 137, 2027) | "Pretrained lightweight open-set IDS" | **No KD** (confirmed from the full PDF, see below) |
| [Soft Labeling Affects OOD Detection](https://arxiv.org/abs/2007.03212) | Student OOD detection tracks the teacher's soft labels | Image domain only. Implies "the student inherits OOD ability" alone is not novel; the traffic study must add the time dimension, shortcuts and the teacher swap. |

Other 2026 papers cover only one side:
- **SepSpace** (Cybersecurity, Aug 2026): open-set, payload-free, prototype-radius rejection; no KD.
- **Multi-teacher KD** (Future Internet 18(4):197, 2026): distillation, no open-set evaluation.

## PreDyn-IDS: full-text reading (Luo et al., Information Fusion 137, 2027)
- **Model:** a 2.1M-parameter pretrained network: a wavelet-attention CNN per packet plus Mamba over the first 8 packets. Pretrained with masked-packet (MAE) and masked-flow (MLM) objectives on ~204k flows from ISCXBot2014, VNAT, ISCXVPN and ISCXTor.
- **Unknown detection:** a confidence branch trained with label "hints" (Eq. 8). This is DeVries & Taylor 2018 (arXiv 1802.04865), not cited, plus an EMA-based dynamic threshold.
- **Data and splits:** Edge-IIoT, ISCXVPN (App/Service), ISCXTor; random 8:1:1 splits.
- **Weaknesses:**
  - **Leakage:** its pretraining pool includes its own test datasets.
  - **Open-world test:** ISCXTor only (6 known, 10 unknown classes), shown as confusion matrices, with no AUROC, FPR or OSCR.
  - **Inputs:** keeps 976 payload bytes; only IPs and ports are anonymised.
  - **Speed:** throughput measured only on an A800 GPU.
  - **Missing:** calibration metrics, XGBoost/k-NN baselines and code.
  - **Inconsistencies:** Mamba-2 in §4.5.1 but first-generation Mamba in §5; a leftover "Packet2Flow" name.
- **Role in this study:** the baseline of a small pretrained open-set model trained directly, alongside NetMamba.

## New evaluation critiques to cite
- [*When Simple Model Just Works: Is Network Traffic Classification in Crisis?*](https://arxiv.org/abs/2506.08655) (CESNET group, 2025): over 50% of samples are duplicated across train and test; k-NN matches complex models.
- [*The Colossus with Feet of Clay*](https://arxiv.org/abs/2608.22683) (Aug 2026): encrypted-traffic classifiers fail under post-quantum TLS evolution.
- Yang et al., TIFS 2025, lightweight and dynamic open-set IDS for industrial IoT (https://doi.org/10.1109/tifs.2025.3546849): no KD.

## Implications for the paper
1. **The contribution** must rest on teacher-specific inheritance (teacher swap), the time dimension and shortcut transfer, not on "the student inherits OOD ability" alone.
2. **Baselines:** NetClus (method); ResAware (prior evidence that inheritance happens); NetMamba and a DeVries-style confidence head (small models trained directly); XGBoost and k-NN.
3. **Before freezing the pre-registration:**
   - run a manual Google Scholar and IEEE Xplore search for KD + unknown detection + CESNET-TLS-Year22;
   - check for new ECH work from the ResAware group.
