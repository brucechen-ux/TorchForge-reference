# Configuration Difference Audit

- A preserves the historical AdamW optimizer behavior, including epsilon `1e-8`.
- B uses report-aligned Muon Hybrid NS: momentum 0.95, Nesterov, weight decay 0.1, RMS target 0.18, Hybrid 8+2, auxiliary AdamW epsilon `1e-20`.
- C is identical to B except `newton_schulz: standard`; both carry the same coefficient blocks so the checker can prove no hidden differences.
- `scripts/check_muon_config_diff.py` passed and ignores only run identity/output paths plus the NS method selector.
