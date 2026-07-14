# Muon Actual Execution Trace

## Historical pre-fix path
1. Launch: `run_experiment.py` emits `torchrun --nproc_per_node 8 -m src.train` commands (`run_plan.json`).
2. Config: `src/train.py:803` loads YAML and applies CLI overrides. Historical B/C resolve to `newton_schulz_iterations: 8`.
3. Model/DDP: `src/train.py:862-871` constructs the model, then wraps native DDP.
4. Grouping: historical `src/train.py:492-512` accepted only `ndim==2` and substring exclusions. Runtime reconstruction found 26,322,112 Muon elements and 371,036,923 AdamW elements; packed expert tensors were 3-D and excluded.
5. Scheduler: `src/train.py:873-879` creates one LambdaLR over the combined optimizer.
6. Backward/sync: `src/train.py:991-1009` uses `no_sync()` only before the final accumulation microstep; final gradients are all-reduced before step.
7. Step: `src/train.py:1041-1047` clips, steps optimizer, steps scheduler, then clears gradients.
8. Historical momentum/Nesterov: old `src/muon.py:221-223` stored a `(1-mu)`-scaled momentum via `lerp`; NS direction was scale-equivalent after normalization but state did not literally match Algorithm 1.
9. Historical NS: old `src/muon.py:54-60` ran only 3 high-slope steps, then cubic-like steps; B/C configs capped total at 8.
10. Historical scaling: old `src/muon.py:261` used `sqrt(max(1, rows/cols))`, with no `gamma=0.18`.
11. Decay/update: old `src/muon.py:267-270` correctly used decoupled decay once, then `-lr*update`.

## Fixed report-aligned path
- Config parsing: `src/train.py:540-557` reads explicit hybrid/standard stage parameters and RMS target.
- Grouping: `src/train.py:499-515` applies semantic AdamW exclusions and sends 2-D plus packed 3-D logical matrices to Muon. Post-fix: 327,926,976 Muon (82.53%), 69,432,059 AdamW (17.47%), zero duplicates/missing.
- Tensor policy: parameters/gradients remain model dtype/device; Frobenius norm is FP32 (`src/muon.py:47-56`); CUDA NS matmuls are BF16 (`src/muon.py:52`).
- Exact recurrence: `src/muon.py:244-245` computes `M=mu*M+G`, `N=mu*M+G`.
- Logical matrices: `src/muon.py:252-278` unbinds packed expert tensors and independently performs NS plus `sqrt(max(rows,cols))*0.18`.
- Final update: `src/muon.py:305-310` applies decoupled decay before the LR-scaled update.
- Checkpoint: plain save/load includes model, optimizer and scheduler state (`src/train.py:128-164`); DeepSpeed delegates state to engine (`src/train.py:604-622`).

## Eight-GPU validation evidence
- Hardware: 8×NVIDIA H800 80GB, CUDA available, BF16 supported on every rank.
- Dedicated NCCL test: rank parameter max error 0; update RMS 0.1799419; BF16-vs-FP32 relative update error 0.01840; momentum error 0 (`audit_results/tests/gpu_nccl_muon_results.json`).
- Hybrid full-model smoke: two steps, real memmap data, world size 8, accumulation 2, 219 Muon tensors, ten NS iterations, peak memory 38.4 GiB/rank (`audit_results/tests/gpu_hybrid_smoke/`).
- Standard full-model smoke: one step under identical initialization/data, 219 Muon tensors, ten Standard iterations (`audit_results/tests/gpu_standard_smoke/`).
