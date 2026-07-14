# Unresolved Official Details

- DeepSeek official Hugging Face repositories expose model cards, configs, weights, encoding and inference code; no complete pretraining optimizer source was found. Therefore this audit claims **report alignment**, not identity with official training code.
- The report does not disclose the Newton–Schulz denominator epsilon, exact overflow-safe Frobenius norm implementation, kernel fusion order, batching thresholds, or per-shape padding details.
- The report states BF16 NS matmuls and stochastic rounding for synchronized MoE gradients, but this CPU-only environment cannot verify the project's GPU BF16 error or stochastic rounding (the project does not implement the latter).
- The exact semantics for project-specific router matrices, packed expert storage and non-mHC scalar parameters are not fully specified by the report; the chosen policy keeps routers/vectors in AdamW and independently orthogonalizes each packed expert matrix.
- No usable project checkpoint was present under `checkpoints/`; module-level effective-scale diagnostics therefore use deterministic synthetic matrices and a small diagnostic smoke batch, not a saved 5B-token state.
