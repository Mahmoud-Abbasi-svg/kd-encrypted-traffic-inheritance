# Exploratory analyses: REVISION_FULL3

Added during revision at the reviewer's request. **Not pre-registered**; the ten confirmatory hypotheses and their numbers are unchanged and were produced by `scripts/08_analyze.py`.

- Windows: `test`; units: 18; bootstrap resamples: 300

## Does the ensemble age faster than its members?

```
score         energy  knnfeat     maha      msp
model                                          
ensemble    -0.00285 -0.00349 -0.00354 -0.00306
member_mean -0.00285 -0.00350 -0.00353 -0.00282
teacherA_0  -0.00278 -0.00340 -0.00350 -0.00282
teacherA_1  -0.00306 -0.00343 -0.00359 -0.00291
teacherA_2  -0.00261 -0.00347 -0.00347 -0.00270
teacherA_3  -0.00289 -0.00370 -0.00370 -0.00283
teacherA_4  -0.00291 -0.00349 -0.00337 -0.00287
```

## Time trend per start date

```
 start  n_windows  slope_per_week    first     last condition  score
    11          9        -0.00049 +0.01527 -0.00844       kdA energy
    24          6        +0.00014 +0.00174 +0.00575       kdA energy
    37          3        -0.00152 +0.00297 -0.00879       kdA energy
    11          9        -0.00044 +0.03178 +0.01695      kdA4 energy
    24          6        -0.00055 +0.03399 +0.02412      kdA4 energy
    37          3        -0.00112 +0.03211 +0.02352      kdA4 energy
```

## Calendar overlap between start dates

```
 start       split weeks  n_weeks  weeks_shared_with_another_start
    11 test_w16-19 16-19        4                                0
    11 test_w20-23 20-23        4                                0
    11 test_w24-27 24-27        4                                0
    11 test_w28-31 28-31        4                                3
    11 test_w32-35 32-35        4                                4
    11 test_w36-39 36-39        4                                4
    11 test_w40-43 40-43        4                                4
    11 test_w44-47 44-47        4                                4
    11 test_w48-51 48-51        4                                4
    24 test_w29-32 29-32        4                                4
    24 test_w33-36 33-36        4                                4
    24 test_w37-40 37-40        4                                4
    24 test_w41-44 41-44        4                                4
    24 test_w45-48 45-48        4                                4
    24 test_w49-51 49-51        3                                3
    37 test_w42-45 42-45        4                                4
    37 test_w46-49 46-49        4                                4
    37 test_w51-51 51-51        1                                1
```

## Chance-level teacher preference

An undistilled student's |ρ difference| between two Teacher A members: median 0.0057, 95th percentile 0.0156 over 540 pairs.

## Teacher-swap shifts

```
                                              label condition        own      other  units  estimate  ci_low  ci_high  p_one_sided  n_boot  units_used
                 H1 replication: kdA4 toward A vs B      kdA4   teacherA   teacherB     18   +0.0071 +0.0046  +0.0091      +0.0033     300          18
                 H1 replication: kdB4 toward B vs A      kdB4   teacherB   teacherA     18   +0.0255 +0.0235  +0.0280      +0.0033     300          18
single vs single: kdM0 toward member 0 vs Teacher B      kdM0 teacherA_0   teacherB     18   +0.0171 +0.0150  +0.0193      +0.0033     300          18
    identity only: kdM0 toward member 0 vs member 1      kdM0 teacherA_0 teacherA_1     18   +0.0224 +0.0207  +0.0242      +0.0033     300          18
    identity only: kdM1 toward member 1 vs member 0      kdM1 teacherA_1 teacherA_0     18   +0.0193 +0.0170  +0.0217      +0.0033     300          18
                        anchor: hardA toward A vs B     hardA   teacherA   teacherB     18   -0.0000 -0.0021  +0.0019      +0.5714     300          18
             family swap: kdC toward Teacher C vs A       kdC   teacherC   teacherA      9   +0.0432 +0.0297  +0.0573      +0.0033     300           9
```

## Detection advantage over the direct student, by scoring rule

```
  score                    comparison    kind  estimate  ci_low  ci_high  p_one_sided  n_boot  units_used  units
 energy     teacherA - direct student teacher   +0.0004 -0.0036  +0.0041      +0.4419     300          18     18
 energy     teacherB - direct student teacher   +0.0051 +0.0012  +0.0090      +0.0066     300          18     18
 energy directTS_w16 - direct student student   -0.0241 -0.0333  -0.0179      +1.0000     300           9      9
 energy directTS_w96 - direct student student   +0.0008 -0.0026  +0.0041      +0.3455     300           9      9
 energy   direct_w16 - direct student student   -0.0240 -0.0332  -0.0178      +1.0000     300           9      9
 energy   direct_w96 - direct student student   +0.0016 -0.0018  +0.0049      +0.2957     300           9      9
 energy        enddA - direct student student   -0.0927 -0.1004  -0.0852      +1.0000     300          18     18
 energy        hardA - direct student student   -0.0027 -0.0056  -0.0005      +0.9900     300          18     18
 energy          kdA - direct student student   -0.0007 -0.0025  +0.0010      +0.7841     300          18     18
 energy         kdA4 - direct student student   -0.0261 -0.0299  -0.0226      +1.0000     300          18     18
 energy     kdA4_w16 - direct student student   -0.0538 -0.0624  -0.0455      +1.0000     300           9      9
 energy     kdA4_w96 - direct student student   -0.0157 -0.0197  -0.0122      +1.0000     300           9      9
 energy          kdB - direct student student   -0.0037 -0.0061  -0.0021      +1.0000     300          18     18
 energy         kdB4 - direct student student   -0.0249 -0.0290  -0.0213      +1.0000     300          18     18
 energy          kdC - direct student student   +0.0043 +0.0009  +0.0076      +0.0100     300           9      9
 energy          kdF - direct student student   +0.0034 -0.0006  +0.0069      +0.0465     300           9      9
 energy         kdM0 - direct student student   -0.0282 -0.0330  -0.0239      +1.0000     300          18     18
 energy         kdM1 - direct student student   -0.0267 -0.0305  -0.0231      +1.0000     300          18     18
 energy           ls - direct student student   -0.0217 -0.0284  -0.0154      +1.0000     300          18     18
    msp     teacherA - direct student teacher   +0.0242 +0.0230  +0.0255      +0.0033     300          18     18
    msp     teacherB - direct student teacher   -0.0071 -0.0089  -0.0055      +1.0000     300          18     18
    msp directTS_w16 - direct student student   -0.0074 -0.0171  +0.0047      +0.9103     300           9      9
    msp directTS_w96 - direct student student   +0.0005 -0.0024  +0.0026      +0.3588     300           9      9
    msp   direct_w16 - direct student student   -0.0076 -0.0173  +0.0045      +0.9203     300           9      9
    msp   direct_w96 - direct student student   -0.0029 -0.0056  -0.0009      +1.0000     300           9      9
    msp        enddA - direct student student   -0.0082 -0.0102  -0.0058      +1.0000     300          18     18
    msp        hardA - direct student student   -0.0025 -0.0037  -0.0011      +1.0000     300          18     18
    msp          kdA - direct student student   -0.0006 -0.0015  +0.0002      +0.9169     300          18     18
    msp         kdA4 - direct student student   -0.0270 -0.0294  -0.0246      +1.0000     300          18     18
    msp     kdA4_w16 - direct student student   -0.0511 -0.0537  -0.0479      +1.0000     300           9      9
    msp     kdA4_w96 - direct student student   -0.0155 -0.0175  -0.0129      +1.0000     300           9      9
    msp          kdB - direct student student   -0.0040 -0.0053  -0.0025      +1.0000     300          18     18
    msp         kdB4 - direct student student   -0.0356 -0.0376  -0.0337      +1.0000     300          18     18
    msp          kdC - direct student student   -0.0308 -0.0330  -0.0289      +1.0000     300           9      9
    msp          kdF - direct student student   +0.0005 -0.0010  +0.0018      +0.2458     300           9      9
    msp         kdM0 - direct student student   -0.0268 -0.0298  -0.0237      +1.0000     300          18     18
    msp         kdM1 - direct student student   -0.0258 -0.0273  -0.0242      +1.0000     300          18     18
    msp           ls - direct student student   -0.0068 -0.0090  -0.0048      +1.0000     300          18     18
   maha     teacherA - direct student teacher   +0.0727 +0.0639  +0.0822      +0.0033     300          18     18
   maha     teacherB - direct student teacher   +0.0320 +0.0230  +0.0415      +0.0033     300          18     18
   maha        enddA - direct student student   +0.0738 +0.0664  +0.0823      +0.0033     300          18     18
   maha        hardA - direct student student   +0.0182 +0.0040  +0.0332      +0.0033     300           9      9
   maha          kdA - direct student student   -0.0031 -0.0100  +0.0055      +0.7243     300          18     18
   maha         kdA4 - direct student student   +0.0763 +0.0691  +0.0833      +0.0033     300          18     18
   maha          kdB - direct student student   +0.0023 -0.0049  +0.0091      +0.2824     300          18     18
   maha         kdB4 - direct student student   +0.0531 +0.0442  +0.0610      +0.0033     300          18     18
   maha          kdF - direct student student   +0.0137 -0.0095  +0.0328      +0.1262     300           9      9
   maha         kdM0 - direct student student   +0.0776 +0.0607  +0.0885      +0.0033     300           9      9
   maha         kdM1 - direct student student   +0.0705 +0.0496  +0.0912      +0.0033     300           9      9
   maha           ls - direct student student   +0.0876 +0.0815  +0.0942      +0.0033     300          18     18
knnfeat     teacherA - direct student teacher   +0.0338 +0.0280  +0.0395      +0.0033     300          18     18
knnfeat     teacherB - direct student teacher   +0.0259 +0.0199  +0.0315      +0.0033     300          18     18
knnfeat        enddA - direct student student   +0.0440 +0.0376  +0.0498      +0.0033     300          18     18
knnfeat        hardA - direct student student   +0.0169 +0.0052  +0.0294      +0.0033     300           9      9
knnfeat          kdA - direct student student   +0.0011 -0.0052  +0.0062      +0.3522     300          18     18
knnfeat         kdA4 - direct student student   +0.0261 +0.0195  +0.0314      +0.0033     300          18     18
knnfeat          kdB - direct student student   +0.0026 -0.0028  +0.0061      +0.2226     300          18     18
knnfeat         kdB4 - direct student student   +0.0086 +0.0027  +0.0136      +0.0033     300          18     18
knnfeat          kdF - direct student student   +0.0176 +0.0006  +0.0279      +0.0100     300           9      9
knnfeat         kdM0 - direct student student   +0.0270 +0.0165  +0.0368      +0.0033     300           9      9
knnfeat         kdM1 - direct student student   +0.0220 +0.0076  +0.0327      +0.0033     300           9      9
knnfeat           ls - direct student student   +0.0426 +0.0373  +0.0483      +0.0033     300          18     18
 energy                     ls - kdA4    pair   +0.0044 -0.0008  +0.0091      +0.0764     300          18     18
    msp                     ls - kdA4    pair   +0.0202 +0.0173  +0.0233      +0.0033     300          18     18
   maha                     ls - kdA4    pair   +0.0113 +0.0080  +0.0146      +0.0033     300          18     18
knnfeat                     ls - kdA4    pair   +0.0166 +0.0126  +0.0200      +0.0033     300          18     18
 energy                   enddA - kdA    pair   -0.0920 -0.0995  -0.0847      +1.0000     300          18     18
    msp                   enddA - kdA    pair   -0.0076 -0.0095  -0.0056      +1.0000     300          18     18
   maha                   enddA - kdA    pair   +0.0769 +0.0728  +0.0810      +0.0033     300          18     18
knnfeat                   enddA - kdA    pair   +0.0429 +0.0384  +0.0473      +0.0033     300          18     18
 energy                    kdA4 - kdA    pair   -0.0254 -0.0290  -0.0221      +1.0000     300          18     18
    msp                    kdA4 - kdA    pair   -0.0264 -0.0293  -0.0239      +1.0000     300          18     18
   maha                    kdA4 - kdA    pair   +0.0794 +0.0752  +0.0834      +0.0033     300          18     18
knnfeat                    kdA4 - kdA    pair   +0.0249 +0.0205  +0.0288      +0.0033     300          18     18
 energy                  hardA - kdA4    pair   +0.0233 +0.0205  +0.0264      +0.0033     300          18     18
    msp                  hardA - kdA4    pair   +0.0246 +0.0228  +0.0263      +0.0033     300          18     18
   maha                  hardA - kdA4    pair   -0.0647 -0.0694  -0.0604      +1.0000     300           9      9
knnfeat                  hardA - kdA4    pair   -0.0115 -0.0183  -0.0043      +1.0000     300           9      9
 energy                    kdF - kdM0    pair   +0.0327 +0.0288  +0.0373      +0.0033     300           9      9
    msp                    kdF - kdM0    pair   +0.0299 +0.0228  +0.0366      +0.0033     300           9      9
   maha                    kdF - kdM0    pair   -0.0639 -0.0710  -0.0531      +1.0000     300           9      9
knnfeat                    kdF - kdM0    pair   -0.0094 -0.0157  -0.0044      +1.0000     300           9      9
 energy                    kdF - kdA4    pair   +0.0304 +0.0277  +0.0328      +0.0033     300           9      9
    msp                    kdF - kdA4    pair   +0.0300 +0.0263  +0.0352      +0.0033     300           9      9
   maha                    kdF - kdA4    pair   -0.0692 -0.0808  -0.0609      +1.0000     300           9      9
knnfeat                    kdF - kdA4    pair   -0.0108 -0.0151  -0.0061      +1.0000     300           9      9
```
