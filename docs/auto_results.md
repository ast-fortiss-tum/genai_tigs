# Auto-generated results (2026-11-27 19:22)

Written by `scripts/auto_summary.py` from `runs/main` (machine snapshot; interpretation is in WORKLOG.md).

## Queue status (last lines of runs/main/status.log)
```
2026-11-27 11:55:32 celebahq dp -> 50 seeds, attempt 1 (have 30)
2026-11-27 16:01:00 celebahq dp exit 0 (have 50/50)
2026-11-27 16:01:00 celebahq dp-paper -> 50 seeds, attempt 1 (have 30)
2026-11-27 16:43:27 queue_100 stopped by hand (replaced by queue_weekend.sh, WORKLOG (23)); celebahq dp-paper run left running
2026-11-27 19:13:32 queue_weekend start
2026-11-27 19:14:08 analysis celebahq fp vs dp written
2026-11-27 19:14:45 analysis celebahq fp vs dp-paper written
2026-11-27 19:16:08 t3_separate cifar10 n50 written
2026-11-27 19:16:28 div_perm cifar10 n50 written
2026-11-27 19:17:57 t3_separate imagenet_pizza n50 written
2026-11-27 19:18:16 div_perm imagenet_pizza n50 written
2026-11-27 19:20:07 t3_separate imagenet_teddy n50 written
2026-11-27 19:20:28 div_perm imagenet_teddy n50 written
2026-11-27 19:22:01 t3_separate celebahq n50 written
2026-11-27 19:22:22 div_perm celebahq n50 written
```

## Seeds per arm

| Task | fp | dp | dp-paper | dp-paper-nostyle |
|---|---|---|---|---|
| cifar10 | 50 | 50 | 50 | 0 |
| imagenet_pizza | 50 | 50 | 50 | 0 |
| imagenet_teddy | 50 | 50 | 50 | 0 |
| celebahq | 50 | 50 | 50 | 0 |

## RQ1-RQ3 (paired: first n seeds common to FP and arm)

validity = seed_ok %, Fisher p; eff = % of valid seeds with a failure (low level), Fisher p; iters = mean iterations to first failure (low / high), Mann-Whitney p.

| Task | Arm | n | validity FP / arm (p) | eff low FP / arm (p) | iters low FP / arm (p) | iters high FP / arm (p) |
|---|---|---|---|---|---|---|
| cifar10 | dp | 50 | 94 / 86 (0.32) | 91 / 88 (0.73) | 72 / 63 (0.36) | 14 / 12 (0.52) |
| cifar10 | dp-paper | 50 | 94 / 90 (0.71) | 91 / 87 (0.52) | 72 / 60 (0.25) | 14 / 12 (0.49) |
| imagenet_pizza | dp | 50 | 78 / 58 (0.053) | 97 / 100 (1) | 74 / 73 (0.89) | 13 / 11 (0.55) |
| imagenet_pizza | dp-paper | 50 | 78 / 86 (0.44) | 97 / 95 (1) | 74 / 121 (0.0056) | 13 / 18 (0.068) |
| imagenet_teddy | dp | 50 | 90 / 54 (0.0001) | 100 / 100 (1) | 54 / 34 (0.091) | 7 / 4 (0.047) |
| imagenet_teddy | dp-paper | 50 | 90 / 94 (0.71) | 100 / 100 (1) | 54 / 38 (0.055) | 7 / 5 (0.017) |
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
