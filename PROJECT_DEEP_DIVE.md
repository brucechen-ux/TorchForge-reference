# DeepSeek-V4 Muon 报告对齐项目深度解析

## 0. 结论与分析边界

本仓库是一个约 397M 参数的缩小训练复现包。它以 `DeepSeekV3TinyLM` 为主模型容器，组合了 V4 风格 attention、V4/V3 混合 MoE、V3 风格 MTP，以及按 DeepSeek-V4 技术报告修正后的 Muon。它不是 DeepSeek-V4 官方完整预训练代码，也没有实现完整的大规模训练系统。

代码中没有 tokenizer 训练或文本编码流程；数据文件已经是 `uint32` token ID。没有找到 FP8、DualPipe、RLVR、QK-Clip、随机舍入、官方 fused/batched Muon 或完整 Hybrid ZeRO 实现。`bf16` 是训练 autocast 开关，不等于 FP8。

仓库内真正可直接运行的受控实验只有三组：AdamW、Muon Hybrid 8+2、Muon Standard 10-step。它们验证的是修正后 Muon 及 Newton-Schulz 策略，不是逐项验证所有 V3/V4 架构差异。

## 1. 项目目录树

```text
.
├── README_PACKAGE_中文.md
├── STRUCTURE.md
├── PROJECT_DEEP_DIVE.md
├── requirements.txt
├── run_three_experiments.sh
├── configs/
│   ├── muon_v4_cpu.yaml
│   ├── rerun_adamw.yaml
│   ├── rerun_muon_standard_ns.yaml
│   └── rerun_v4_muon_hybrid.yaml
├── src/
│   ├── __init__.py
│   ├── checkpoint_io.py
│   ├── data.py
│   ├── eval.py
│   ├── infer_benchmark.py
│   ├── long_context_eval.py
│   ├── metrics.py
│   ├── mla.py
│   ├── modeling_v3.py
│   ├── moe.py
│   ├── mtp.py
│   ├── muon.py
│   ├── profile_inference.py
│   ├── train.py
│   ├── utils.py
│   └── v4_attention.py
├── scripts/
│   ├── check_env.py
│   ├── check_muon_config_diff.py
│   └── check_torch.py
├── tests/
│   ├── ddp_consistency_test.py
│   ├── gpu_nccl_muon_test.py
│   └── run_audit_tests.py
├── audit_results/
│   ├── commands/
│   ├── patches/deepseek_v4_muon_alignment.diff
│   ├── deepseek_v4_audit_report.md
│   ├── deepseek_v4_compliance_matrix.csv
│   ├── optimizer_parameter_groups.csv
│   ├── muon_execution_trace.md
│   ├── ns_numerical_tests.csv
│   ├── one_step_reference_check.csv
│   ├── update_scale_analysis.csv
│   └── 其他审计说明与 CSV
└── reports/
    └── previous_muon_experiment_brief_zh.docx
```

### 1.1 目录作用

| 目录 | 作用 |
|---|---|
| `configs/` | 模型、V4 attention、MoE、MTP、训练、优化器和数据的完整运行配置。 |
| `src/` | 训练入口、模型结构、attention/MoE/MTP/Muon、数据、评估、checkpoint 和推理工具。 |
| `scripts/` | 环境检查和 B/C 配置等价性检查，不参与训练前向。 |
| `tests/` | CPU 数值审计、2-rank DDP 一致性、8-GPU NCCL/BF16 Muon 单步验证。 |
| `audit_results/` | 历史实现问题、修复 diff、参数分组、数值测试和建议重跑计划的证据包。 |
| `reports/` | 上一轮实验 Word 简报，非程序输入。 |

## 2. 程序入口与训练总流程

### 2.1 入口

- 单次训练入口：`src/train.py::main`，通过 `python -m src.train --config <yaml>` 调用。
- 三实验入口：`run_three_experiments.sh`，依次启动 A/B/C 三次 8-rank `torchrun`。
- 离线评估入口：`src/eval.py::main`。
- 长上下文评估入口：`src/long_context_eval.py::main`。
- 推理 profiling：`src/profile_inference.py::main`。
- `src/infer_benchmark.py` 只是兼容转发到 `profile_inference.main`。

### 2.2 完整训练调用链

```text
run_three_experiments.sh / python -m src.train
  ↓
src.train.main()
  ├─ parse_args()
  ├─ load_yaml() → apply_overrides()
  ├─ seed_everything() → init_distributed()
  ├─ 根据 global_batch_tokens 与 target_tokens 解析 max_steps
  ├─ build_dataloaders()
  │   ├─ build_datasets()
  │   ├─ SyntheticTokenDataset 或 MemmapTokenDataset
  │   ├─ DistributedSampler（world_size > 1）
  │   └─ DataLoader → {input_ids, labels}
  ├─ DeepSeekV3LikeLM(config) == DeepSeekV3TinyLM(config)
  ├─ DDP(...) 或 initialize_deepspeed(...)
  ├─ build_optimizer()
  │   ├─ InstrumentedAdamW
  │   └─ MuonWithAuxAdamW（Muon 矩阵组 + AdamW 辅助组）
  ├─ build_scheduler() → warmup + cosine + min_lr_ratio
  ├─ 可选 load_checkpoint()/load_deepspeed_checkpoint()
  └─ step loop
      ├─ gradient accumulation micro-step loop
      │   ├─ next(DataLoader) → move_batch_to_device()
      │   ├─ BF16 autocast（仅 native 路径）
      │   ├─ model(input_ids, labels)
      │   ├─ loss / accumulation_steps
      │   └─ backward；非末 micro-step 使用 DDP.no_sync()
      ├─ apply_pending_balance_bias_updates()
      ├─ clip_grad_norm_()
      ├─ optimizer.step()
      ├─ scheduler.step()
      ├─ optimizer.zero_grad()
      ├─ evaluate()（valid_steps 或最终 step）
      ├─ save_checkpoint()（save_steps）
      └─ JSONL/CSV/TensorBoard/NS diagnostics 日志
```

### 2.3 数据与 tokenizer

`src/data.py::MemmapTokenDataset` 读取 `manifest.json`，再以 NumPy memmap 打开 `train.bin`/`valid.bin`。每个样本取连续 `seq_len + 1` 个 token，前 `seq_len` 个作为 `input_ids`，后 `seq_len` 个作为 next-token `labels`。

```text
原始文本（仓库外完成 tokenizer）
  ↓
uint32 train.bin / valid.bin + manifest.json
  ↓ MemmapTokenDataset.__getitem__
[t0 ... tN]
  ├─ input_ids = [t0 ... tN-1]
  └─ labels    = [t1 ... tN]
  ↓ DataLoader / DistributedSampler
[batch, seq]
  ↓ Embedding → Transformer → LM head
[batch, seq, vocab]
  ↓ cross_entropy(ignore_index=-100)
LM loss
```

仓库没有 tokenizer 类、词表加载、BPE/SentencePiece 或 raw-text preprocessing。YAML 中的 `vocab_size=49152` 只约束 token ID 空间；`build_datasets()` 还会检查 manifest 与配置的词表大小一致。

### 2.4 scheduler、checkpoint、evaluation

- Scheduler：`src/utils.py::build_scheduler`，线性 warmup 后 cosine decay，最低比例为 `min_lr / learning_rate`；Muon 与辅助 AdamW 是同一个 optimizer 的参数组，因此共享 scheduler。
- Native checkpoint：`src/train.py::save_checkpoint/load_checkpoint` 保存模型、optimizer、scheduler、step 和 config。
- DeepSpeed checkpoint：`save_deepspeed_checkpoint/load_deepspeed_checkpoint` 委托 engine 保存，并在 `client_state` 中保留 step/config。
- 评估：`src/train.py::evaluate` 聚合 `loss/lm_loss/mtp_loss/aux_loss`，分布式时 all-reduce 总和与计数。
- 独立评估：`src/eval.py` 通过 `src/checkpoint_io.py` 同时兼容普通 `.pt` 和 DeepSpeed checkpoint。

## 3. 模型结构与前向数据流

### 3.1 模型构建

`src/modeling_v3.py::DeepSeekV3TinyLM.__init__` 是唯一实际模型构造器，末尾别名 `DeepSeekV3LikeLM = DeepSeekV3TinyLM` 被训练、评估和 checkpoint 工具引用。

```text
DeepSeekV3TinyLM
├─ embed_tokens
├─ DeepseekV4RotaryEmbedding
├─ layers: N × TransformerBlock
│  ├─ RMSNorm
│  ├─ DeepseekV4Attention
│  ├─ residual
│  ├─ RMSNorm
│  ├─ DenseSwiGLUFFN 或 DeepSeekV3MoE
│  └─ residual
├─ final_norm
├─ lm_head
└─ mtp_modules: D × DeepSeekV3MTPModule（可选）
```

### 3.2 完整前向链

```text
input_ids
  ↓ embed_tokens
hidden_states
  ├─ position_ids
  ├─ main/compress RoPE
  └─ sliding-window causal mask
  ↓ for TransformerBlock
RMSNorm → DeepseekV4Attention → residual
  ↓
RMSNorm → Dense FFN / Hash-MoE / Learned MoE → residual
  ↓
final_norm → lm_head → logits → LM cross entropy
  ↓ optional MTP chain
MTP shifted labels → shared embedding → fusion → TransformerBlock → shared lm_head
  ↓
total_loss = lm_loss + mtp_loss_weight × mtp_loss + mean(aux_losses)
```

## 4. V3 保留、V4 新增与混合边界

### 4.1 V3 保留

| 模块 | 代码依据 | 保留内容 |
|---|---|---|
| 主容器 | `src/modeling_v3.py::DeepSeekV3TinyLM` | pre-norm Transformer、残差、共享 embedding/LM head、V3 命名和配置兼容。 |
| MLA 参考实现 | `src/mla.py::DeepSeekV3MLA` | V3 低秩 Q/KV 压缩 attention；当前主模型不实例化它，仅被保留作 legacy/reference。 |
| V3 Router | `src/moe.py::DeepSeekV3Gate` | sigmoid/softmax、group-limited top-k、correction bias、无辅助损失均衡机制。 |
| V3 MoE 包装 | `src/moe.py::DeepSeekV3MoE` | shared experts + routed experts + optional aux loss；内部可切换到 V4 router/packed experts。 |
| MTP | `src/mtp.py::DeepSeekV3MTPModule` | shifted future token、embedding/hidden fusion、额外 block 和共享 lm_head。 |
| DeepSpeed MoE | `DeepSeekV3MoEDispatchGate`, `DeepSeekV3DeepSpeedMoE` | V3 grouped routing 到 DeepSpeed dispatch/combine tensor 的适配。 |

### 4.2 V4 新增/替换

| 模块 | 文件与入口 | 相对 V3 的变化 | 设计目的 |
|---|---|---|---|
| V4 Attention | `src/v4_attention.py::DeepseekV4Attention.forward` | 主路径不再调用 `DeepSeekV3MLA`；加入 HCA/CSA/sliding layer types、压缩器、indexer、partial interleaved RoPE、Q/KV unweighted RMSNorm、output low-rank/grouped projection。 | 用压缩 KV 与稀疏索引降低长序列 attention 成本，并稳定 QK 数值。 |
| HCA/CSA | `DeepseekV4HCACompressor`, `DeepseekV4CSACompressor`, `DeepseekV4Indexer` | 根据 layer type 选择 heavily compressed、compressed sparse 或 sliding attention。 | 在不同层分配不同压缩率和稀疏检索模式。 |
| V4 top-k router | `src/moe.py::DeepSeekV4TopKRouter` | 支持 `sqrtsoftplus` affinity；可叠加 correction/balance bias。 | 复现 V4 风格 affinity 与动态负载均衡。 |
| Hash-MoE | `DeepSeekV4HashRouter` | token ID 通过固定 `tid2eid` 直接选专家，权重仍来自 learned affinity。 | 前若干层提供确定性路由 bootstrap。 |
| Packed experts | `DeepSeekV4Experts` | 每组专家以 3-D `gate_up_proj/down_proj` 存储；可切到 `DeepSeekV4ModuleListExperts`。 | 接近 HF V4 权重布局并提高参数/执行组织性。 |
| SwiGLU clipping | `SwiGLUExpert`, `DeepSeekV4Experts._apply_gate` | 对 gate/up 激活做 `swiglu_limit` 截断。 | 控制极端激活。 |
| Muon | `src/muon.py::MuonWithAuxAdamW` | 2-D/packed 3-D 逻辑矩阵走 Muon；其他参数走内部 AdamW。 | 对矩阵更新正交化，同时保留向量、norm、embedding 等参数的 AdamW 稳定性。 |

### 4.3 名称不能等同于官方代际

当前模型类名仍是 V3，但 attention 已完全替换成 V4 路径；MoE 外壳叫 V3，内部默认却使用 V4 router 和 packed experts。因此应把本仓库理解为混合复现，不应仅凭类名判断代际。审计报告也明确只声称“与公开报告对齐”，不声称与官方训练代码相同。

## 5. 重要模块逐项定位

### 5.1 MLA / V4 Attention

- 文件：`src/mla.py`（V3 reference）、`src/v4_attention.py`（实际主路径）、`src/modeling_v3.py`（构建与调用）。
- 入口：`TransformerBlock.__init__ → DeepseekV4Attention`；前向为 `TransformerBlock.forward → DeepseekV4Attention.forward`。
- 配置：`v4_attention.*`；`_normalize_v4_attention_config` 也能从 legacy `mla.*` 读取部分字段。
- 开关：配置中虽有 `v4_attention.enabled`，但构造代码没有读取该字段；当前主模型实际上始终构造 V4 attention。这不是一个有效的 V3/V4 attention runtime 开关。
- 底层链：Q LoRA 投影 → Q norm；KV projection → KV norm；partial RoPE；按 layer type 压缩/index；SDPA/eager/可选 flash local attention；grouped low-rank output projection。

### 5.2 MoE、Router、Experts

- 文件：`src/moe.py`，构建点 `TransformerBlock.__init__`。
- 入口：`DeepSeekV3MoE.forward`。
- 调用链：`TransformerBlock.forward → DeepSeekV3MoE.forward → gate/router → shared_experts + routed experts → load/aux stats`。
- 配置：`num_routed_experts`, `num_shared_experts`, `num_experts_per_token`, `num_hash_layers`, `mlp_layer_types`, `num_expert_groups`, `num_limited_groups`, `route_scale`, `score_function`, `swiglu_limit`, `use_packed_experts`, `implementation`, `moe_ep_size`, capacity/drop/Tutel 参数。
- 开关边界：`moe.enabled` 没有在模型构建中读取；真正控制 dense/MoE 的是 `first_dense_layers` 和 `mlp_layer_types`。当前实验 `first_dense_layers=0`，默认前三层 Hash-MoE，后续 learned MoE。

### 5.3 Aux Loss / No Aux Loss / Load Balance / Dynamic Bias

- Aux loss：`DeepSeekV3MoE.forward` 在 `aux_loss_weight > 0` 时计算 expert importance × detached load；总 loss 在 `DeepSeekV3TinyLM.forward` 中直接加上 mean aux loss。
- No Aux Loss：当前三个正式 YAML 都设 `aux_loss_weight: 0.0`，因此不向反向图加入辅助均衡损失。
- Correction bias：`use_correction_bias` 创建 non-parameter buffer `balance_bias` 并在选 top-k 前加到 route score；实际 mixture 权重仍从未加 bias 的原 score 取得。
- Dynamic bias：只有 `balance_bias: true` 时，`stage_balance_bias_update` 才记录 expert counts；训练入口在一次 forward/backward 后调用 `apply_pending_balance_bias_updates` 更新并 clamp。
- 当前正式配置没有显式 `balance_bias: true`，normalize 默认值为 false，所以虽然 `use_correction_bias: true` 创建了零 bias buffer，但不会动态变化。
- bias 可通过 `--load_balance_bias`、`--save_balance_bias` 和 `--balance_bias_calibration` 单独校准、保存和恢复。

### 5.4 MTP

- 文件：`src/mtp.py`、`src/modeling_v3.py`。
- 入口：`DeepSeekV3MTPModule.forward`。
- 调用链：主干 hidden → shifted future labels → shared embedding → `enorm/hnorm` → concat → `eh_proj` → TransformerBlock → shared lm_head → MTP CE loss。
- 配置：`enabled`, `mtp_depth`, `mtp_loss_weight`, `mtp_use_moe`。
- `mtp_use_moe=false` 时额外 MTP block 被强制为 dense FFN；true 时沿用 MoE。
- 三个正式配置都启用 depth=1、weight=0.1，因此它不是 A/B/C 的变量。

### 5.5 Muon / AdamW

- 文件：`src/muon.py`；参数分组和构建在 `src/train.py::build_optimizer`。
- AdamW 入口：`InstrumentedAdamW.step`，算法仍由 `torch.optim.AdamW` 完成，仅增加指标采样。
- Muon 入口：`MuonWithAuxAdamW.step`。
- Muon 调用链：梯度 → momentum `M=mu*M+G` → Nesterov `N=mu*M+G` → 每个逻辑矩阵 `newton_schulz` → `sqrt(max(rows, cols))*0.18` → decoupled weight decay → `param.add_(update, alpha=-lr)`。
- Hybrid：前 8 次系数 `(3.4445,-4.775,2.0315)`，后 2 次 `(2,-1.5,0.5)`。
- Standard：10 次 `(2,-1.5,0.5)`。
- 分组：2-D 和 3-D 矩阵默认进入 Muon；embedding、LM head、norm、router、FFN gate、balance/static bias、gating factor 和用户额外排除项进入辅助 AdamW。
- 审计证据：修复后 327,926,976 参数（82.53%）进 Muon，69,432,059（17.47%）进 AdamW；无重复或遗漏。

### 5.6 FP8、DualPipe、RLVR

全仓库没有这些实现或配置入口：

- FP8：无 FP8 dtype、scale、amax、recipe 或通信量化代码。
- DualPipe：无 pipeline stage、双向流水调度或 bubble 计算。
- RLVR：无 rollout、reward/verifier、policy/reference model、PPO/GRPO 或 RL loss。

这些属于 DeepSeek 系列背景技术，但不是本项目的代码能力。

## 6. 全部实验与消融

### 6.1 正式 A/B/C

| 实验 | 配置/脚本 | 修改模块与参数 | 验证目的 |
|---|---|---|---|
| A: AdamW baseline | `configs/rerun_adamw.yaml`; `run_three_experiments.sh` 第 1 项 | `optimizer.name=adamw`, betas=(0.9,0.95), eps=1e-8；其他模型/数据与 B/C 一致。 | 建立历史 AdamW 基线，比较相同 LR、token budget 下的收敛。 |
| B: V4 Muon Hybrid | `configs/rerun_v4_muon_hybrid.yaml`; 脚本第 2 项 | Muon, momentum=.95, Nesterov, Hybrid 8+2, RMS target=.18, aux AdamW eps=1e-20。 | 验证报告对齐 Muon 是否优于/追平 AdamW，以及修复历史错误后的真实效果。 |
| C: Muon Standard NS | `configs/rerun_muon_standard_ns.yaml`; 脚本第 3 项 | 与 B 仅 `newton_schulz: standard` 不同；10 次 standard 系数。 | 隔离 Newton-Schulz 系数策略，比较 Hybrid 8+2 与稳定但收敛较慢的 Standard。 |

`scripts/check_muon_config_diff.py` 会把 run name/output path/NS selector 归一化后比较 B/C，保证没有隐藏变量。

### 6.2 Smoke 与数值消融

| 实验/检查 | 文件 | 验证内容 |
|---|---|---|
| CPU smoke | `configs/muon_v4_cpu.yaml` | 小模型 synthetic data，两步 Muon 路径、日志和诊断可运行；MTP 关闭。 |
| NS 数值比较 | `tests/run_audit_tests.py`, `audit_results/ns_numerical_tests.csv` | historical、Hybrid、Standard 在多 shape/condition 下的正交误差和 polar approximation。 |
| RMS scaling | 同上；`update_scale_analysis.csv` | 不同矩阵形状缩放后 update RMS 接近 0.18。 |
| Nesterov/decay 单步 | 同上；`one_step_reference_check.csv` | momentum、Nesterov、decoupled decay 和最终参数 delta 与 reference 一致。 |
| 参数分组 | 同上；`optimizer_parameter_groups.csv` | 每个 trainable parameter 是否唯一进入正确 optimizer group。 |
| 2-rank DDP | `tests/ddp_consistency_test.py` | Gloo 同步后不同 rank 参数一致。 |
| 8-GPU NCCL/BF16 | `tests/gpu_nccl_muon_test.py` | rank 一致性、RMS、momentum、BF16-vs-FP32 update error。 |
| Full-model smoke | `audit_results/commands/gpu_validation_commands.txt` | Hybrid 两步、Standard 一步，在真实 memmap/8×H800 下无 NaN/Inf。 |

### 6.3 存在开关但没有独立实验配置的消融面

- `mtp.enabled`, `mtp_use_moe`, `mtp_depth`, `mtp_loss_weight`
- `aux_loss_weight` 与 `balance_bias`
- `use_packed_experts`
- `score_function`: sigmoid/softmax/sqrtsoftplus
- `mlp_layer_types`: dense/hash_moe/moe
- `implementation`: torch/deepspeed
- `moe_ep_size`, capacity/drop/Tutel
- attention `layer_types`, compress rates, sliding window, index top-k
- activation checkpointing、BF16、attention compile/implementation

这些开关可由 YAML 或部分 CLI override 改变，但仓库没有对应成套结果，不能把它们写成已验证结论。

### 6.4 历史结果的正确解读

审计报告记录历史 validation loss：A=2.950770、B=3.001184、C=3.043975。但历史 Muon 存在三个关键错误：Hybrid NS 不是报告的 8+2、缺失 RMS=0.18 缩放、3-D packed expert 被错误送入 AdamW。因此历史 A/B/C 不能证明 Muon 弱于 AdamW；本包的目标是修复后重跑，而不是把旧结果当成最终消融结论。

## 7. 关键底层调用关系

### 7.1 Attention

```text
DeepSeekV3TinyLM.forward
→ TransformerBlock.forward
→ RMSNorm(attn input)
→ DeepseekV4Attention.forward
  → q_a_proj → q_a_norm → q_b_proj
  → kv_proj → kv_norm
  → apply_rotary_pos_emb
  → HCACompressor / CSACompressor / sliding path
  → Indexer → IndexerScorer → top-k compressed blocks（CSA）
  → SDPA / eager_attention_forward / flash_local_attention_forward
  → o_a_proj → o_b_proj
→ residual add
```

### 7.2 MoE

```text
TransformerBlock.forward
→ ffn_norm
→ DeepSeekV3MoE.forward
  → HashRouter 或 TopKRouter
    → affinity logits
    → score_function
    → optional correction bias
    → top-k indices + original-score weights
  → shared_experts(SwiGLU)
  → DeepSeekV4Experts / ModuleListExperts
    → token-expert dispatch mask
    → gate/up projection → clipped SwiGLU → down projection
    → weighted index_add
  → stage balance-bias counts
  → optional aux loss + routing stats
→ residual add
```

### 7.3 Loss 与 backward

```text
lm_head logits → CE(labels) = lm_loss
MTP modules → mean CE = mtp_loss
MoE layers → mean weighted balance loss = aux_loss
total = lm + mtp_weight*mtp + aux
→ total / grad_accum_steps
→ backward
→ DDP final micro-step all-reduce
→ clip grad
→ optimizer step
```

### 7.4 Muon

```text
build_optimizer
→ semantic parameter classification
→ Muon group (2-D/3-D logical matrices) + AdamW group
→ MuonWithAuxAdamW.step
  → exact momentum/Nesterov
  → unbind packed expert dimension（3-D）
  → stable FP32 Frobenius normalization
  → BF16 CUDA NS matmul / FP32 fallback
  → Hybrid 8+2 or Standard 10
  → RMS target scaling
  → stack logical updates
  → decoupled weight decay once
  → LR-scaled parameter update
```

## 8. 配置开关索引

| 领域 | 关键配置 | 读取位置 |
|---|---|---|
| Model | vocab/seq/layers/hidden/heads/dense size/tied embeddings | `DeepSeekV3TinyLM.__init__` |
| Attention | layer types, q rank, head dims, RoPE, compress rates, sliding, index, implementation | `_normalize_v4_attention_config` |
| MoE | experts/top-k/hash layers/groups/score/bias/packed/DS capacity | `_normalize_moe_config`, `_normalize_mlp_layer_types` |
| MTP | enabled/depth/loss weight/use MoE | `DeepSeekV3TinyLM.__init__/forward` |
| Optimizer | AdamW/Muon, NS method/stages/coefficients/RMS/momentum/eps | `src.train.build_optimizer` |
| Train | steps/tokens/batch/accum/LR/warmup/BF16/save/valid | `src.train.main` |
| Data | synthetic/memmap, files, workers, pin/prefetch | `src.data` |
| Distributed | native DDP or CLI DeepSpeed config, EP size | `src.train.main`, `src.moe` |

## 9. 关键代码索引

| 功能 | 文件 | 类/函数 |
|---|---|---|
| 训练入口 | `src/train.py` | `main`, `parse_args`, `apply_overrides` |
| 模型构建 | `src/modeling_v3.py` | `DeepSeekV3TinyLM`, `TransformerBlock` |
| 模型前向/loss | `src/modeling_v3.py` | `DeepSeekV3TinyLM.forward` |
| V4 attention | `src/v4_attention.py` | `DeepseekV4Attention`, `DeepseekV4Attention.forward` |
| 压缩 attention | `src/v4_attention.py` | `DeepseekV4HCACompressor`, `DeepseekV4CSACompressor`, `DeepseekV4Indexer` |
| V3 MLA reference | `src/mla.py` | `DeepSeekV3MLA` |
| MoE 总入口 | `src/moe.py` | `DeepSeekV3MoE.forward` |
| V4 router | `src/moe.py` | `DeepSeekV4TopKRouter`, `DeepSeekV4HashRouter` |
| V3 grouped router | `src/moe.py` | `DeepSeekV3Gate` |
| Experts | `src/moe.py` | `DeepSeekV4Experts`, `DeepSeekV4ModuleListExperts`, `SwiGLUExpert` |
| DeepSpeed MoE | `src/moe.py` | `DeepSeekV3MoEDispatchGate`, `DeepSeekV3DeepSpeedMoE` |
| MTP | `src/mtp.py` | `DeepSeekV3MTPModule.forward` |
| Muon | `src/muon.py` | `newton_schulz`, `MuonWithAuxAdamW.step` |
| AdamW instrumentation | `src/muon.py` | `InstrumentedAdamW.step` |
| 参数分组 | `src/train.py` | `build_optimizer` |
| 数据 | `src/data.py` | `MemmapTokenDataset`, `SyntheticTokenDataset`, `build_dataloaders` |
| Scheduler | `src/utils.py` | `build_scheduler` |
| Checkpoint | `src/train.py`, `src/checkpoint_io.py` | save/load 与通用 checkpoint loader |
| 训练内评估 | `src/train.py` | `evaluate` |
| 独立评估 | `src/eval.py` | `main` |
| 长上下文评估 | `src/long_context_eval.py` | `main` |
| 推理 profiling | `src/profile_inference.py` | `main` |

## 10. 风险与限制

1. `v4_attention.enabled` 和 `moe.enabled` 出现在 YAML，但当前构造器不读取，不能视作有效总开关。
2. `src/mla.py` 的 V3 MLA 不在训练主路径；做 V3 attention baseline 需要显式改造构造器，而不是只改 YAML。
3. 正式 YAML 使用 native DDP。普通 DeepSpeed/ZeRO 可能切分 Muon 所需完整逻辑矩阵，审计材料明确未验证该组合。
4. V4 generation cache 仍未接入主模型；`use_cache`/`past_key_values` 会抛出 `NotImplementedError`。
5. 当前 `torch` MoE 逐专家 Python 循环适合研究复现，不代表官方大规模 fused kernel 性能。
6. 没有原始数据、checkpoint 或本次完整重跑日志，代码包只能证明实现与 smoke/audit 状态，不能提供修复后三组最终收敛结论。
7. 官方未公开 NS epsilon、融合顺序、padding/batching 等细节，所以“报告对齐”不等于“官方实现一致”。
