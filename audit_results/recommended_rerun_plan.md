# Recommended Rerun Plan

## Acceptance status
- CPU unit tests, parameter uniqueness, exact Nesterov, RMS scaling, weight decay, one-step reference, config diff and 2-rank Gloo DDP consistency pass.
- Eight-GPU H800 NCCL/BF16 validation is complete: dedicated single-step reference PASS, full Hybrid two-step smoke PASS, full Standard one-step smoke PASS.

## Controlled experiments
1. A: `configs/rerun_adamw.yaml` — unchanged AdamW baseline behavior.
2. B: `configs/rerun_v4_muon_hybrid.yaml` — report-aligned Hybrid 8+2 using the existing project LR.
3. C: `configs/rerun_muon_standard_ns.yaml` — identical to B except ten standard stabilizing NS iterations.

GPU smoke has passed. Do not change learning rate for the first aligned rerun. After report-aligned results exist, run only short sweeps at 0.5x, 0.75x, 1.0x, 1.5x and 2.0x the AdamW LR.
