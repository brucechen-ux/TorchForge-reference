# DeepSeek-V4 Muon Implementation Compliance Audit

## Scope and evidence

This audit used the official DeepSeek-V4 technical report linked by the official model card (`arXiv:2606.19348`), specifically §2.4 / Algorithm 1 (PDF p.14), §3.4.1 (pp.19–20), §4.2.2 (p.25), and query/KV normalization in §2.3.3 (p.12). The four official Hugging Face repositories were checked; they expose model cards, configs, weights, encoding and inference code, but no complete pretraining optimizer implementation was found. Conclusions therefore mean **aligned with disclosed report behavior**, not “identical to official training code.”

Historical experiment results are A=2.950770, B=3.001184 and C=3.043975 validation loss (`analysis_results/main_results.csv`). No complete project checkpoint was available in `checkpoints/`. GPU validation was subsequently completed on eight NVIDIA H800 80GB GPUs with CUDA BF16 and NCCL.

## Executive verdict

The implementation used for the historical B/C runs was **not a compliant DeepSeek-V4 Muon implementation**. It had three critical mismatches sufficient to invalidate a direct optimizer-quality conclusion:

1. Hybrid NS executed eight total iterations, but only the first three used the V4 high-slope coefficients; the remaining five used a different cubic update. The official final two stabilizing quintic iterations did not exist.
2. Muon updates were not rescaled to RMS 0.18. The code used `sqrt(max(1, rows/cols))` and omitted `gamma=0.18`.
3. Parameter grouping sent all packed 3-D expert weights to AdamW. Runtime reconstruction found only 26.32M/397.36M elements in Muon; after correction, 327.93M elements are in Muon.

These issues plausibly explain Muon underperforming AdamW and mean the historical B/C comparison does not isolate the intended algorithms.

## Required questions

1. **Was current historical Muon scaled to RMS=0.18?** No. Historical code lacked gamma and the required `sqrt(max(n,m))`. The repaired implementation passes all tested shapes with relative RMS error below 0.00023 (`audit_results/update_scale_analysis.csv`).
2. **Did historical Hybrid NS run 8 or 10 times?** Eight total calls, configured at `experiment.yaml:38` and `configs_resolved/B_muon_v4_ns.yaml`; not ten.
3. **Did the final two stabilizing iterations exist?** No. Historical code switched after three iterations to `1.5X-0.5XXᵀX`, not two `(2,-1.5,0.5)` quintic steps. Fixed at `src/muon.py:61`.
4. **Did grouping comply?** No historically. Packed experts were 3-D and went to AdamW; generic `gate` exclusion also over-excluded semantic projection matrices. Fixed grouping is at `src/train.py:499`.
5. **Were embedding, LM head and RMSNorm on AdamW?** Yes historically and after repair. The semantic exclusions are verified parameter-by-parameter in `audit_results/optimizer_parameter_groups.csv`.
6. **Was Nesterov input correct?** Historical direction was scale-equivalent after Frobenius normalization, but the stored buffer and literal formula differed due to `lerp`. The repair exactly computes `M=mu*M+G`, then `N=mu*M+G` at `src/muon.py:244`; the one-step test is exact.
7. **Was weight decay correct and once?** Yes. It was and remains decoupled, applied once before `-lr*update` (`src/muon.py:305`).
8. **Did Muon and auxiliary AdamW share the scheduler?** Yes. They are parameter groups of one optimizer and one LambdaLR (`src/train.py:891`; step at `src/train.py:1064`).
9. **Was same numeric LR based on correct RMS scaling?** No historically. Therefore “same LR” did not have the report's intended meaning. The new B/C configs establish RMS 0.18 before reusing the AdamW LR.
10. **Top three likely causes of worse Muon loss:** missing RMS=0.18 scaling; incorrect/short Hybrid NS; most expert weights optimized by AdamW rather than Muon. The historical AdamW epsilon `1e-8` vs report `1e-20` is secondary.
11. **Algorithm errors vs system differences:** NS stages, RMS scaling, grouping and literal Nesterov state were algorithm mismatches. Lack of hybrid ZeRO, fused/batched NS, stochastic rounding and DeepSeek communication kernels are system/performance differences unless they split logical matrices or alter numerics. Native DDP is algorithmically valid because complete gradients are synchronized before NS.
12. **Safe to rerun now?** Yes, with the generated native-DDP configurations. The 8-GPU NCCL/BF16 single-step reference, two-step full Hybrid smoke and one-step full Standard smoke all passed. Do not switch these runs to the unverified DeepSpeed/ZeRO path.

## Parameter grouping

Post-fix totals from the full 397M model:

- Muon: 327,926,976 elements (82.53%).
- AdamW: 69,432,059 elements (17.47%).
- Duplicate trainable parameters: 0.
- Missing trainable parameters: 0.
- Report-policy mismatches: 0 under the documented semantic policy.
- Tied weights: the rerun model is untied; the test uses parameter identity, so a tied parameter cannot be assigned twice.

The report does not explicitly classify this project's non-mHC routers. They remain in AdamW as a conservative project policy and are documented as an interpretation, not an official requirement.

## Numerical findings

- Report Hybrid 8+2 sharply improves difficult square/conditioned test cases relative to the historical implementation. Example condition-1e3 polar relative error: historical 0.3059 vs repaired Hybrid 0.000222.
- Standard ten-step `(2,-1.5,0.5)` is intentionally slower-converging on some square cases; this is a valid controlled coefficient-strategy comparison, not evidence to reduce its iteration count.
- Zero and below-epsilon matrices remain finite and near zero rather than becoming arbitrary orthogonal matrices. This is safe and expected for a zero update, though their RMS cannot be 0.18.
- Large-norm overflow was removed with an FP32 scale-stable Frobenius norm. The report does not disclose its exact epsilon/overflow implementation.
- Exact Nesterov, decoupled decay and final one-step weight update all match the reference with zero observed FP32 error.
- Two-rank Gloo test produced `ddp_max_abs_error=0`.
- Eight-rank NCCL test produced rank parameter error 0, scaled update RMS 0.179942 (relative error 0.000323), momentum error 0, and BF16-vs-FP32 update error 1.84%.
- Full 397M Hybrid smoke ran two steps on 8×H800 with 219 Muon parameter tensors, ten NS iterations, no NaN/Inf and peak memory about 38.4 GiB/rank.
- Full Standard smoke ran one step with the same initial loss/data and 219 Muon tensors; diagnostics confirmed ten Standard iterations.

## Learning-rate audit

The project has one learning rate in each resolved config and no extra global-batch LR scaling. Tokens per step are `micro_batch_size * accumulation * sequence_length * world_size`; this affects schedule duration via target-token-derived max steps, not LR magnitude. Warmup, cosine decay and minimum LR are shared across optimizer groups. Plain checkpoints save optimizer and scheduler state; DeepSpeed delegates them to the engine.

Do not infer that Muon needs a larger LR from the historical runs. First run A/B/C with the corrected mechanism and existing AdamW LR. Only afterward use short sweeps at 0.5x, 0.75x, 1.0x, 1.5x and 2.0x.

## Architecture and QK-Clip

The project applies RMSNorm to attention queries and KV immediately before attention (`src/v4_attention.py:770`–`776`), matching the architectural condition described in §2.3.3 and §2.4. No QK-Clip implementation was found. Because the project is a reduced/custom V4-like architecture, independent attention-logit monitoring is still required; this finding does not justify blindly changing other architectures.

## Acceptance table

| 检查项 | 状态 | 严重程度 | 证据 | 是否阻止重跑 |
| --- | --- | --- | --- | --- |
| 参数分组 | PASS after critical fix | CRITICAL | `optimizer_parameter_groups.csv`; pre-fix only 6.62% Muon | No, fixed |
| Nesterov | PASS after exact-form fix | HIGH | `one_step_reference_check.csv` | No |
| Hybrid NS 8+2 | PASS after critical fix | CRITICAL | `src/muon.py:61`; `ns_numerical_tests.csv` | No, fixed |
| RMS=0.18 | PASS after critical fix | CRITICAL | `update_scale_analysis.csv` | No, fixed |
| Weight decay | PASS | HIGH | one-step exact delta | No |
| Scheduler | PASS | HIGH | one optimizer / one LambdaLR | No |
| DDP correctness | PASS | HIGH | 2-rank Gloo exact; 8-rank NCCL parameter error 0 | No |
| 数值稳定性 | PASS | MEDIUM | BF16-vs-FP32 update error 1.84%; full Hybrid/Standard smoke pass | No |

## Final status

**READY_FOR_RERUN**

The critical algorithmic mismatches are repaired and all stated pre-rerun acceptance conditions for the native-DDP path now pass. The three controlled experiments may be launched with the generated configs. Do not change learning rate for the first aligned comparison and do not use the unverified DeepSpeed/ZeRO Muon path.
