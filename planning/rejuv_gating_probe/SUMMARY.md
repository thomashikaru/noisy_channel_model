# Rejuv gating signal probe -- summary

Signal: 2329 inputs; ops evaluated: 1591 (1498 repairs, 1258 stimuli with an edit); skipped: 0

## Window w0 (gate within +/-0 units of the repair site)

tau | op recall | repair recall (all ops) | stim recall (any repair) | grid kept | items w/ any spike
---|---|---|---|---|---
-2 | 0.899 | 0.906 | 0.902 | 0.675 | 1.000
-1 | 0.799 | 0.814 | 0.814 | 0.500 | 1.000
0 | 0.622 | 0.634 | 0.668 | 0.348 | 1.000
1 | 0.384 | 0.393 | 0.448 | 0.247 | 0.998
2 | 0.233 | 0.239 | 0.283 | 0.134 | 0.972
3 | 0.118 | 0.123 | 0.147 | 0.079 | 0.712
4 | 0.060 | 0.063 | 0.076 | 0.025 | 0.238
5 | 0.041 | 0.043 | 0.051 | 0.007 | 0.074
6 | 0.026 | 0.027 | 0.033 | 0.004 | 0.046
8 | 0.011 | 0.012 | 0.014 | 0.001 | 0.019
10 | 0.004 | 0.005 | 0.006 | 0.001 | 0.012

## Window w1 (gate within +/-1 units of the repair site)

tau | op recall | repair recall (all ops) | stim recall (any repair) | grid kept | items w/ any spike
---|---|---|---|---|---
-2 | 0.992 | 0.992 | 0.990 | 0.941 | 1.000
-1 | 0.935 | 0.940 | 0.931 | 0.793 | 1.000
0 | 0.843 | 0.852 | 0.850 | 0.616 | 1.000
1 | 0.700 | 0.712 | 0.756 | 0.477 | 0.998
2 | 0.427 | 0.437 | 0.517 | 0.287 | 0.972
3 | 0.270 | 0.280 | 0.334 | 0.169 | 0.712
4 | 0.120 | 0.125 | 0.149 | 0.057 | 0.238
5 | 0.052 | 0.053 | 0.064 | 0.018 | 0.074
6 | 0.029 | 0.030 | 0.036 | 0.010 | 0.046
8 | 0.012 | 0.013 | 0.015 | 0.004 | 0.019
10 | 0.004 | 0.005 | 0.006 | 0.003 | 0.012

## Window w2 (gate within +/-2 units of the repair site)

tau | op recall | repair recall (all ops) | stim recall (any repair) | grid kept | items w/ any spike
---|---|---|---|---|---
-2 | 0.997 | 0.997 | 0.997 | 0.990 | 1.000
-1 | 0.966 | 0.967 | 0.963 | 0.918 | 1.000
0 | 0.882 | 0.888 | 0.890 | 0.785 | 1.000
1 | 0.753 | 0.761 | 0.810 | 0.653 | 0.998
2 | 0.451 | 0.460 | 0.543 | 0.410 | 0.972
3 | 0.280 | 0.291 | 0.347 | 0.246 | 0.712
4 | 0.124 | 0.129 | 0.153 | 0.085 | 0.238
5 | 0.052 | 0.054 | 0.064 | 0.027 | 0.074
6 | 0.030 | 0.031 | 0.037 | 0.016 | 0.046
8 | 0.012 | 0.013 | 0.015 | 0.007 | 0.019
10 | 0.004 | 0.005 | 0.006 | 0.004 | 0.012

## Top-k policy, window w0 (site within +/-0 of a top-k unit)

k | op recall | repair recall (all ops) | stim recall (any repair) | grid kept
---|---|---|---|---
1 | 0.080 | 0.084 | 0.100 | 0.090
2 | 0.292 | 0.298 | 0.355 | 0.179
3 | 0.450 | 0.457 | 0.527 | 0.269
4 | 0.598 | 0.599 | 0.657 | 0.358
6 | 0.839 | 0.840 | 0.847 | 0.538
8 | 0.944 | 0.941 | 0.940 | 0.709

## Top-k policy, window w1 (site within +/-1 of a top-k unit)

k | op recall | repair recall (all ops) | stim recall (any repair) | grid kept
---|---|---|---|---
1 | 0.244 | 0.255 | 0.304 | 0.192
2 | 0.576 | 0.579 | 0.689 | 0.391
3 | 0.776 | 0.773 | 0.817 | 0.537
4 | 0.884 | 0.879 | 0.895 | 0.667
6 | 0.974 | 0.972 | 0.967 | 0.853
8 | 0.997 | 0.997 | 0.997 | 0.953

## Top-k policy, window w2 (site within +/-2 of a top-k unit)

k | op recall | repair recall (all ops) | stim recall (any repair) | grid kept
---|---|---|---|---
1 | 0.258 | 0.269 | 0.320 | 0.290
2 | 0.651 | 0.640 | 0.762 | 0.562
3 | 0.845 | 0.836 | 0.886 | 0.733
4 | 0.937 | 0.933 | 0.955 | 0.856
6 | 0.997 | 0.997 | 0.997 | 0.963
8 | 0.999 | 0.999 | 0.999 | 0.994

## Per-dataset op recall at w1: threshold (tau = 0 / 2 / 4) and top-k (k = 2 / 4)

dataset | n_ops | tau=0 | tau=2 | tau=4 | k=2 | k=4 | mean units/item
---|---|---|---|---|---|---|---
tabor2004 | 64 | 0.953 | 0.344 | 0.016 | 0.672 | 0.953 | 16.0
huang2024 | 73 | 0.753 | 0.027 | 0.000 | 0.137 | 0.726 | 15.5
gibson2013 | 122 | 0.689 | 0.410 | 0.057 | 0.451 | 0.664 | 9.5
clark2026 | 144 | 0.812 | 0.486 | 0.326 | 0.688 | 0.896 | 12.9
qian2023 | 480 | 0.933 | 0.163 | 0.015 | 0.452 | 0.896 | 12.1
chen2023 | 330 | 0.603 | 0.258 | 0.018 | 0.385 | 0.833 | 8.8
ryskin2021 | 378 | 1.000 | 0.984 | 0.325 | 0.966 | 1.000 | 12.1

## Op recall at w1 by op type (indel grid serves ins/del; sub-sweep serves replace)

op_tag | n | tau=0 | tau=2 | k=2 | k=4
---|---|---|---|---|---
insert | 323 | 0.889 | 0.421 | 0.551 | 0.941
delete | 211 | 0.379 | 0.081 | 0.223 | 0.578
replace | 1057 | 0.922 | 0.498 | 0.654 | 0.928

## Item-level gate: any unit with rel >= tau (whole-item skip candidate)

Inputs needing an edit: 1254 / 2329

tau | fires on edit-needing items | fires on clean items
---|---|---
-2 | 1.000 | 1.000
-1 | 1.000 | 1.000
0 | 1.000 | 1.000
1 | 0.998 | 0.997
2 | 0.979 | 0.964
3 | 0.749 | 0.670
4 | 0.280 | 0.189
5 | 0.110 | 0.032
6 | 0.069 | 0.019
8 | 0.027 | 0.010
10 | 0.015 | 0.007

## Where does the item's biggest spike sit relative to the repair site?

d(item argmax, site): -5: 93, -4: 12, -3: 13, -2: 2, -1: 12, +0: 127, +1: 249, +2: 20, +3: 144, +4: 265, +5: 654  (clamped to +/-5)

