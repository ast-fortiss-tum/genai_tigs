# Auto-generated results (2026-11-28 07:29)

Written by `scripts/auto_summary.py` from `runs/main` (machine snapshot; interpretation is in WORKLOG.md).

## Queue status (last lines of runs/main/status.log)
```
2026-11-27 19:22:26 cifar10 dp-paper-nostyle -> 50 seeds, attempt 1 (have 0)
2026-11-28 01:37:40 cifar10 dp-paper-nostyle exit 0 (have 50/50)
2026-11-28 01:38:11 analysis cifar10 fp vs dp-paper-nostyle written
2026-11-28 01:40:09 t3_separate cifar10 n50 written
2026-11-28 01:40:35 div_perm cifar10 n50 written
2026-11-28 01:40:35 imagenet_pizza dp-paper-nostyle -> 50 seeds, attempt 1 (have 0)
2026-11-28 05:25:15 imagenet_pizza dp-paper-nostyle exit 0 (have 50/50)
2026-11-28 05:26:01 analysis imagenet_pizza fp vs dp-paper-nostyle written
2026-11-28 05:28:33 t3_separate imagenet_pizza n50 written
2026-11-28 05:28:58 div_perm imagenet_pizza n50 written
2026-11-28 05:28:58 imagenet_teddy dp-paper-nostyle -> 50 seeds, attempt 1 (have 0)
2026-11-28 07:26:17 imagenet_teddy dp-paper-nostyle exit 0 (have 50/50)
2026-11-28 07:26:53 analysis imagenet_teddy fp vs dp-paper-nostyle written
2026-11-28 07:28:41 t3_separate imagenet_teddy n50 written
2026-11-28 07:29:07 div_perm imagenet_teddy n50 written
```

## Seeds per arm

| Task | fp | dp | dp-paper | dp-paper-nostyle |
|---|---|---|---|---|
| cifar10 | 50 | 50 | 50 | 50 |
| imagenet_pizza | 50 | 50 | 50 | 50 |
| imagenet_teddy | 50 | 50 | 50 | 50 |
| celebahq | 50 | 50 | 50 | 0 |

## RQ1-RQ3 (paired: first n seeds common to FP and arm)

validity = seed_ok %, Fisher p; eff = % of valid seeds with a failure (low level), Fisher p; iters = mean iterations to first failure (low / high), Mann-Whitney p.

| Task | Arm | n | validity FP / arm (p) | eff low FP / arm (p) | iters low FP / arm (p) | iters high FP / arm (p) |
|---|---|---|---|---|---|---|
| cifar10 | dp | 50 | 94 / 86 (0.32) | 91 / 88 (0.73) | 72 / 63 (0.36) | 14 / 12 (0.52) |
| cifar10 | dp-paper | 50 | 94 / 90 (0.71) | 91 / 87 (0.52) | 72 / 60 (0.25) | 14 / 12 (0.49) |
| cifar10 | dp-paper-nostyle | 50 | 94 / 84 (0.2) | 91 / 88 (0.73) | 72 / 63 (0.56) | 14 / 13 (0.73) |
| imagenet_pizza | dp | 50 | 78 / 58 (0.053) | 97 / 100 (1) | 74 / 73 (0.89) | 13 / 11 (0.55) |
| imagenet_pizza | dp-paper | 50 | 78 / 86 (0.44) | 97 / 95 (1) | 74 / 121 (0.0056) | 13 / 18 (0.068) |
| imagenet_pizza | dp-paper-nostyle | 50 | 78 / 58 (0.053) | 97 / 100 (1) | 74 / 73 (0.89) | 13 / 11 (0.55) |
| imagenet_teddy | dp | 50 | 90 / 54 (0.0001) | 100 / 100 (1) | 54 / 34 (0.091) | 7 / 4 (0.047) |
| imagenet_teddy | dp-paper | 50 | 90 / 94 (0.71) | 100 / 100 (1) | 54 / 38 (0.055) | 7 / 5 (0.017) |
| imagenet_teddy | dp-paper-nostyle | 50 | 90 / 64 (0.0037) | 100 / 100 (1) | 54 / 36 (0.083) | 7 / 4 (0.047) |
| celebahq | dp | 50 | 98 / 98 (1) | 71 / 71 (1) | 102 / 112 (0.49) | 28 / 26 (0.82) |
| celebahq | dp-paper | 50 | 98 / 98 (1) | 71 / 69 (1) | 102 / 98 (0.76) | 28 / 25 (0.63) |

## RQ6 diversity, pre-registered test (WORKLOG (23))

Eq. 6 on the classifier input, failure images; diff = D_arm - D_FP; permutation p (10k), 95% bootstrap CI, Holm per arm and feature space over the task x level tests listed. **Primary = ResNet-50.**

| Arm | Feature | Task | Level | n seeds | D FP | D arm | diff [95% CI] | p | p Holm |
|---|---|---|---|---|---|---|---|---|---|
| dp | resnet50 | cifar10 | low | 50 | 0.526 | 0.541 | +0.015 [-0.021, +0.053] | 0.391 | 1 |
| dp | resnet50 | cifar10 | high | 50 | 0.527 | 0.537 | +0.010 [-0.030, +0.050] | 0.631 | 1 |
| dp | resnet50 | imagenet_pizza | low | 50 | 0.347 | 0.366 | +0.019 [-0.055, +0.083] | 0.594 | 1 |
| dp | resnet50 | imagenet_pizza | high | 50 | 0.399 | 0.417 | +0.018 [-0.061, +0.100] | 0.675 | 1 |
| dp | resnet50 | imagenet_teddy | low | 50 | 0.544 | 0.558 | +0.014 [-0.036, +0.061] | 0.601 | 1 |
| dp | resnet50 | imagenet_teddy | high | 50 | 0.562 | 0.564 | +0.002 [-0.046, +0.050] | 0.936 | 1 |
| dp | resnet50 | celebahq | low | 50 | 0.624 | 0.664 | +0.041 [-0.076, +0.157] | 0.51 | 1 |
| dp | resnet50 | celebahq | high | 50 | 0.634 | 0.653 | +0.019 [-0.059, +0.099] | 0.631 | 1 |
| dp | clip | cifar10 | low | 50 | 0.225 | 0.222 | -0.004 [-0.029, +0.022] | 0.791 | 1 |
| dp | clip | cifar10 | high | 50 | 0.227 | 0.232 | +0.005 [-0.023, +0.034] | 0.744 | 1 |
| dp | clip | imagenet_pizza | low | 50 | 0.140 | 0.134 | -0.006 [-0.026, +0.014] | 0.574 | 1 |
| dp | clip | imagenet_pizza | high | 50 | 0.167 | 0.161 | -0.006 [-0.044, +0.037] | 0.783 | 1 |
| dp | clip | imagenet_teddy | low | 50 | 0.233 | 0.243 | +0.010 [-0.023, +0.044] | 0.58 | 1 |
| dp | clip | imagenet_teddy | high | 50 | 0.240 | 0.253 | +0.013 [-0.019, +0.045] | 0.449 | 1 |
| dp | clip | celebahq | low | 50 | 0.396 | 0.376 | -0.020 [-0.051, +0.013] | 0.193 | 1 |
| dp | clip | celebahq | high | 50 | 0.388 | 0.390 | +0.002 [-0.025, +0.027] | 0.89 | 1 |
| dp-paper | resnet50 | cifar10 | low | 50 | 0.526 | 0.533 | +0.007 [-0.027, +0.045] | 0.681 | 1 |
| dp-paper | resnet50 | cifar10 | high | 50 | 0.527 | 0.548 | +0.020 [-0.022, +0.059] | 0.325 | 1 |
| dp-paper | resnet50 | imagenet_pizza | low | 50 | 0.347 | 0.498 | +0.151 [+0.057, +0.248] | 0.0067 | 0.0536 |
| dp-paper | resnet50 | imagenet_pizza | high | 50 | 0.399 | 0.525 | +0.126 [+0.027, +0.219] | 0.0178 | 0.0942 |
| dp-paper | resnet50 | imagenet_teddy | low | 50 | 0.544 | 0.596 | +0.051 [+0.007, +0.096] | 0.0157 | 0.0942 |
| dp-paper | resnet50 | imagenet_teddy | high | 50 | 0.562 | 0.621 | +0.059 [+0.013, +0.106] | 0.0124 | 0.0868 |
| dp-paper | resnet50 | celebahq | low | 50 | 0.624 | 0.632 | +0.008 [-0.108, +0.133] | 0.896 | 1 |
| dp-paper | resnet50 | celebahq | high | 50 | 0.634 | 0.635 | +0.002 [-0.082, +0.088] | 0.975 | 1 |
| dp-paper | clip | cifar10 | low | 50 | 0.225 | 0.225 | -0.000 [-0.028, +0.028] | 0.989 | 1 |
| dp-paper | clip | cifar10 | high | 50 | 0.227 | 0.233 | +0.006 [-0.019, +0.030] | 0.619 | 1 |
| dp-paper | clip | imagenet_pizza | low | 50 | 0.140 | 0.223 | +0.083 [+0.038, +0.127] | 0.0012 | 0.0096 ** |
| dp-paper | clip | imagenet_pizza | high | 50 | 0.167 | 0.227 | +0.060 [+0.013, +0.104] | 0.0182 | 0.127 |
| dp-paper | clip | imagenet_teddy | low | 50 | 0.233 | 0.239 | +0.006 [-0.024, +0.040] | 0.724 | 1 |
| dp-paper | clip | imagenet_teddy | high | 50 | 0.240 | 0.253 | +0.013 [-0.016, +0.044] | 0.41 | 1 |
| dp-paper | clip | celebahq | low | 50 | 0.396 | 0.388 | -0.008 [-0.039, +0.023] | 0.574 | 1 |
| dp-paper | clip | celebahq | high | 50 | 0.388 | 0.382 | -0.007 [-0.035, +0.020] | 0.628 | 1 |
| dp-paper-nostyle | resnet50 | cifar10 | low | 50 | 0.526 | 0.544 | +0.018 [-0.016, +0.054] | 0.326 | 1 |
| dp-paper-nostyle | resnet50 | cifar10 | high | 50 | 0.527 | 0.542 | +0.015 [-0.025, +0.055] | 0.477 | 1 |
| dp-paper-nostyle | resnet50 | imagenet_pizza | low | 50 | 0.347 | 0.366 | +0.019 [-0.055, +0.083] | 0.594 | 1 |
| dp-paper-nostyle | resnet50 | imagenet_pizza | high | 50 | 0.399 | 0.417 | +0.018 [-0.061, +0.100] | 0.675 | 1 |
| dp-paper-nostyle | resnet50 | imagenet_teddy | low | 50 | 0.544 | 0.571 | +0.027 [-0.018, +0.072] | 0.264 | 1 |
| dp-paper-nostyle | resnet50 | imagenet_teddy | high | 50 | 0.562 | 0.572 | +0.010 [-0.038, +0.056] | 0.685 | 1 |
| dp-paper-nostyle | clip | cifar10 | low | 50 | 0.225 | 0.218 | -0.008 [-0.034, +0.019] | 0.573 | 1 |
| dp-paper-nostyle | clip | cifar10 | high | 50 | 0.227 | 0.228 | +0.002 [-0.025, +0.028] | 0.91 | 1 |
| dp-paper-nostyle | clip | imagenet_pizza | low | 50 | 0.140 | 0.134 | -0.006 [-0.026, +0.014] | 0.574 | 1 |
| dp-paper-nostyle | clip | imagenet_pizza | high | 50 | 0.167 | 0.161 | -0.006 [-0.044, +0.037] | 0.783 | 1 |
| dp-paper-nostyle | clip | imagenet_teddy | low | 50 | 0.233 | 0.250 | +0.017 [-0.015, +0.047] | 0.316 | 1 |
| dp-paper-nostyle | clip | imagenet_teddy | high | 50 | 0.240 | 0.250 | +0.010 [-0.020, +0.040] | 0.527 | 1 |
