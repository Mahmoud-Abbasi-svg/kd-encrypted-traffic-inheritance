# Analysis: val_all_with_shortcut

- Windows: `val`; bootstrap resamples: 300; seed 2022
- Runs: `20260917-110225_S_train11-14` (start 11), `20260918-121700_S_train11-14` (start 11), `20260917-121305_S_train24-27` (start 24), `20260918-124618_S_train24-27` (start 24), `20260918-132337_S_train37-40` (start 37)
- Shortcut run: `20260918-234156_S_train11-14`
- **Not a confirmatory result:** `val` splits are not the pre-registered test windows.

## Hypotheses (primary analysis, Holm-corrected)

| hypothesis | p_iut | p_holm | supported | note |
|---|---|---|---|---|
| H1 | 0.003322 | 0.02326 | True |  |
| H2[energy] | 0.9801 | 1 | False |  |
| H2[msp] | 0.9701 | 1 | False |  |
| H3[energy] |  |  | False | undefined: one time point |
| H3[msp] |  |  | False | undefined: one time point |
| H4a | 0.9193 | 1 | False |  |
| H4b | 0.9984 | 1 | False |  |
| H5[energy] | 1 | 1 | False |  |
| H5[msp] | 0.003322 | 0.02326 | True |  |

## Components

| analysis | hypothesis | component | estimate | ci_low | ci_high | p_one_sided |
|---|---|---|---|---|---|---|
| primary | H1 | kdA4_shift_to_A | 0.04614 | 0.0393 | 0.05108 | 0.003322 |
| primary | H1 | kdB4_shift_to_B | 0.04167 | 0.03537 | 0.04875 | 0.003322 |
| primary | info | kdA_shift_to_A | 0.003026 | 0.0002234 | 0.005339 |  |
| primary | info | kdB_shift_to_B | -0.002732 | -0.004468 | -0.0009829 |  |
| primary | H2[energy] | auroc_kdA_minus_ls | -0.01137 | -0.02605 | -0.001126 | 0.9801 |
| primary | H2[energy] | auroc_kdA_minus_directTS | -0.002106 | -0.006747 | 0.002885 | 0.7475 |
| primary | H2[energy] | nll_ls_minus_kdA | 0.02434 | 0.02197 | 0.02719 | 0.003322 |
| primary | H2[energy] | nll_directTS_minus_kdA | 0.0002671 | -0.001684 | 0.002436 | 0.4153 |
| primary | H3[energy] | gap_slope_per_week |  |  |  |  |
| primary | H5[energy] | auroc_enddA_minus_kdA | -0.05582 | -0.07357 | -0.03619 | 1 |
| primary | info | mean_gap_teacherA_minus_kdA_energy | 0.02782 | 0.01527 | 0.04467 |  |
| primary | H2[msp] | auroc_kdA_minus_ls | -0.003667 | -0.01003 | 0.003294 | 0.8738 |
| primary | H2[msp] | auroc_kdA_minus_directTS | -0.003147 | -0.006298 | 5.455e-05 | 0.9701 |
| primary | H2[msp] | nll_ls_minus_kdA | 0.02434 | 0.02197 | 0.02719 | 0.003322 |
| primary | H2[msp] | nll_directTS_minus_kdA | 0.0002671 | -0.001684 | 0.002436 | 0.4153 |
| primary | H3[msp] | gap_slope_per_week |  |  |  |  |
| primary | H5[msp] | auroc_enddA_minus_kdA | 0.007559 | 0.001099 | 0.01491 | 0.003322 |
| primary | info | mean_gap_teacherA_minus_kdA_msp | 0.04411 | 0.03899 | 0.04874 |  |
| primary | info | raw_kdA_own_minus_other | 0.03536 | 0.0287 | 0.04212 |  |
| primary | info | raw_kdB_own_minus_other | -0.03506 | -0.04139 | -0.02892 |  |
| primary | info | f1_kdA_minus_ls | 0.008614 | 0.007609 | 0.009858 |  |
| primary | info | f1_kdA_minus_directTS | 0.0006649 | -0.0004945 | 0.001806 |  |
| primary | info | ece_ls_minus_kdA | -0.000171 | -0.001885 | 0.001347 |  |
| primary | info | ece_directTS_minus_kdA | -0.0003383 | -0.0009274 | 0.0003373 |  |
| primary | H4a | reliance_kd_minus_direct | -0.0319 | -0.07102 |  | 0.9193 |
| primary | H4b | ece_minus_rho0 | 0.008593 | 0.006102 |  | 0.0004731 |
| primary | H4b | rho0_minus_auroc | -0.02006 | -0.0277 |  | 0.9984 |

## Mean over units (all flows, all unknowns)

| condition | macro_f1_mean | auroc_energy_mean | auroc_msp_mean | nll_mean | ece_mean | ece_ts_mean |
|---|---|---|---|---|---|---|
| direct | 0.9234 | 0.8525 | 0.883 | 0.1843 | 0.002704 | 0.005495 |
| directTS | 0.9234 | 0.8517 | 0.8836 | 0.1832 | 0.005495 |  |
| enddA | 0.9165 | 0.7937 | 0.8873 | 0.2181 | 0.006637 | 0.004661 |
| kdA | 0.9241 | 0.8495 | 0.8817 | 0.1844 | 0.002858 | 0.005834 |
| kdA4 | 0.9014 | 0.8338 | 0.8674 | 0.2744 | 0.03199 | 0.005092 |
| kdB | 0.9231 | 0.8499 | 0.8821 | 0.1857 | 0.003206 | 0.005766 |
| kdB4 | 0.9033 | 0.8378 | 0.8632 | 0.3021 | 0.03554 | 0.004473 |
| ls | 0.9155 | 0.861 | 0.8834 | 0.2513 | 0.05879 | 0.005642 |
| teacherA | 0.9666 | 0.8773 | 0.9305 | 0.08212 | 0.002089 | 0.003323 |
| teacherA_member | 0.9617 | 0.8614 | 0.9112 | 0.09829 | 0.006495 | 0.002628 |
| teacherB | 0.9657 | 0.8787 | 0.9153 | 0.09235 | 0.006936 | 0.002889 |

## Figures

![fig_gap.png](fig_gap.png)
![fig_specificity.png](fig_specificity.png)
![fig_reliance.png](fig_reliance.png)
