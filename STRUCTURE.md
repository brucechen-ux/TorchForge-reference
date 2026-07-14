# 目录结构

```text
deepseek_v4_muon_report_aligned_package_20260713/
├── README_PACKAGE_中文.md       # 包说明、运行方法和注意事项
├── STRUCTURE.md                 # 本目录结构说明
├── requirements.txt             # Python 依赖
├── run_three_experiments.sh      # A→B→C 八卡 native-DDP 顺序启动脚本
├── src/                          # 模型、训练入口和修复后的 Muon 实现
│   ├── muon.py                   # Muon、Hybrid/Standard NS、RMS scaling、Nesterov
│   ├── train.py                  # 参数分组、optimizer/scheduler、DDP 训练流程
│   ├── modeling_v3.py            # 主模型结构
│   ├── v4_attention.py           # V4 attention 与 Q/KV normalization
│   ├── moe.py                    # MoE、packed experts、router
│   ├── mtp.py                    # MTP 模块
│   ├── data.py                   # 数据加载
│   └── ...                       # eval、checkpoint、metrics 等支持代码
├── configs/
│   ├── rerun_adamw.yaml          # A：AdamW baseline
│   ├── rerun_v4_muon_hybrid.yaml # B：报告对齐 Hybrid 8+2
│   ├── rerun_muon_standard_ns.yaml # C：10 次 Standard NS
│   └── muon_v4_cpu.yaml          # 小模型 CPU smoke 配置
├── scripts/
│   ├── check_muon_config_diff.py # 检查 B/C 除 NS method 外无额外差异
│   ├── check_env.py              # 环境检查
│   └── check_torch.py            # PyTorch/CUDA 检查
├── tests/
│   ├── run_audit_tests.py        # NS、RMS、Nesterov、weight decay、参数分组测试
│   ├── ddp_consistency_test.py   # 两 rank DDP 一致性测试
│   └── gpu_nccl_muon_test.py     # 八卡 NCCL/BF16 Muon 单步测试
├── audit_results/                # 完整审计报告、CSV 证据和统一 diff
│   ├── deepseek_v4_audit_report.md
│   ├── deepseek_v4_compliance_matrix.csv
│   ├── optimizer_parameter_groups.csv
│   ├── muon_execution_trace.md
│   ├── patches/deepseek_v4_muon_alignment.diff
│   └── commands/                 # 审计、smoke 和建议训练命令
└── reports/
    └── previous_muon_experiment_brief_zh.docx # 上一轮结果与问题原因 Word 简报
```
