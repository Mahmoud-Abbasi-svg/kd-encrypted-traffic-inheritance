# Novelty re-check, 19 September 2026

Second scan before freezing the pre-registration (the first was 15 Sep). Focus: work appearing since about June 2026, plus older work the first scan may have missed. Automated search of arXiv, Semantic Scholar, ACM DL abstracts, OpenReview and conference pages; **Google Scholar, IEEE Xplore and dblp block automated access** and still need a manual pass (see "Manual checks still needed").

## Verdict
**The gap is still open.** No work found combines inheritance of unknown-traffic detection, a teacher-swap control on per-flow scores, a months-long time axis, and encrypted traffic. Every single ingredient now has 2026 literature, so several papers move from optional to must-cite.

## Closest work, by claim

### Calibration and "what actually transfers" (our RQ1)
- **Beyond Dark Knowledge: Mixup-Based Distillation for Reliable Predictions**, arXiv:2606.12171 (Jun 2026). Reports that calibration propagates from teacher to student independently of accuracy, and that temperature governs an accuracy–calibration trade-off. **The closest pre-emption of our temperature finding (D8).** Generic ML: no open-set scoring, no teacher swap, no traffic, no time. Our separation of *score-pattern transfer* (happens, teacher-specific) from *detection quality* (does not) is what remains ours.
- **Trust the Uncertain Teacher (CUD)**, arXiv:2602.12687 (Feb, rev. May 2026). Method for making students inherit calibrated uncertainty; assumes transfer is achievable.
- **A Functional Perspective on Knowledge Distillation**, arXiv:2510.12615 (rev. Mar 2026). Large study: functional transfer is weaker than assumed; KD reframed as a data-dependent regulariser. Our most important conceptual antecedent; no OOD, calibration, shortcuts or teacher identity.
- **Knowledge Distillation Must Account for What It Loses**, arXiv:2604.25110 (Apr 2026). Position paper arguing students should be judged on preserved teacher *capabilities*. Our paper is evidence for its thesis; reviewers will expect engagement.
- Older, required: Stanton et al., *Does KD Really Work?* (arXiv:2106.05945, fidelity baseline for our per-flow correlation); Ojha et al., *What Knowledge Gets Distilled* (arXiv:2205.16004); *Teacher's Pet* (bias inheritance).

### Teacher identity, our teacher swap (RQ1, strongest novelty)
- **Who Taught You That? Tracing Teachers in Model Distillation**, arXiv:2502.06659 (ACL 2025). Identifies a student's teacher from its outputs (NLP, lexical fingerprints), with no directly-trained control. Same question, different purpose: forensics rather than a control for a behavioural property.
- **KD Detection for Open-weights Models**, arXiv:2510.02302 (NeurIPS 2025). Same forensic family.
- LLM-provenance work (Antidistillation Fingerprinting, TokenPrint, AttnDiff) shows the idea is in the air.

### Shortcuts (RQ3)
- **Bias in the Shadows (BiasSeeker)**, arXiv:2601.10180 (Jan 2026). Shortcut features in encrypted traffic classification across 19 datasets. No distillation, no model-size comparison, no calibration, no open-set. **Our domain anchor; related work should lead with it.**
- **Anti-Shortcut Distillation (ASD)**, arXiv:2608.11789 (Aug 2026) and **SA-OPD**, arXiv:2608.03632 (Aug 2026). Both assume shortcut transfer through KD and *mitigate* it, in vision and LLMs. We *measure* it, including the condition where the student never sees the feature.
- **Preventing Shortcut Learning in Medical Imaging through Intermediate-Layer KD**, arXiv:2511.17421. Synthetic injected bias features: the closest method lineage to our flip test.
- **Hooker et al., Characterising Bias in Compressed Models**, arXiv:2010.03058, plus arXiv:2205.10828 and arXiv:2110.08419. Prior art for our finding that small students lean on shortcuts about twice as hard. A reviewer will raise Hooker immediately.

### Encrypted traffic and open-world evaluation (RQ2, domain)
- No paper citing CESNET-TLS-Year22 (13 found) touches distillation, calibration or open-set recognition.
- **Open-World Darknet Traffic Recognition under Leave-One-Service-Out**, arXiv:2608.04167 (Aug 2026): open-world traffic with rejection, no KD, no time axis.
- **SoK: Where Do Flow Labels Come From?**, arXiv:2609.02140 (Sep 2026): label provenance; relevant to how we define unknown services.
- **Is NTC in Crisis?**, arXiv:2506.08655; **Fine-grained TLS classification with reject option**, arXiv:2202.11984 (in-domain open-set baseline).
- 2026 KD-for-traffic papers (ResAware 2606.17462, CipherSight 2608.13905, Pruned Traffic Trees, NetClus, MERLOT, a TNSM federated-KD paper, Future Internet 18(4):197) are all about accuracy and efficiency. **None evaluates unknown detection, calibration, shortcuts, teacher identity or drift in the student.**

## Contrary evidence to address in the paper
Two of our findings run against published results and need a defensive paragraph each:
1. **Label smoothing matching or beating KD at unknown detection.** The literature says label smoothing *degrades* OOD detection: arXiv:2007.03212, arXiv:2102.05131, arXiv:2410.06134, arXiv:2403.14715; plus the KD ≈ LS equivalence line (Yuan et al., CVPR 2020). We must explain why our setting differs: ~100 fine-grained classes, flow-level features, MSP vs energy scoring.
2. **Students not inheriting unknown-detection quality.** Several papers report distilled students matching or beating teacher ensembles on OOD: arXiv:2305.10384, arXiv:2203.08295, arXiv:2503.11339, arXiv:2511.13766, arXiv:2206.02152, arXiv:2302.11874. Frame our negative result as regime-specific (single teacher, 101k student, real drift), not universal.

## Manual checks still needed
- **NTC-R 2026** (Network Traffic Classification Reloaded), co-located with ACM CoNEXT, 7–11 Dec 2026: topics include shortcut learning, bias detection and negative results. Acceptances were decided 30 Aug 2026 but **the programme is only published on 15 Oct 2026**. This is the highest scoop risk and cannot be checked yet. Recheck after 15 Oct, or email the organisers.
- **IEEE Xplore:** ETKD, SD-MKD, "Efficient ETC with Multiple Knowledge Distillation" (doc 11227284) and the TNSM 2026 KD + federated paper were judged from abstracts only. Skim for any open-set or calibration evaluation.
- **dblp:** blocked. Worth listing the Luxemburk / Hynek / Čejka 2026 output by hand; that group is the most likely to do our RQ2 first.
- **IMC 2026** cycle-1 acceptances were not verifiable.
