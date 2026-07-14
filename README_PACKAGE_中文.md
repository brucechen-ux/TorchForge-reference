# DeepSeek-V4 Muon 报告对齐代码包

## 包的用途

这是当前约 397M 参数项目的 DeepSeek-V4 Muon 报告对齐版本。核心修复包括：

- Hybrid Newton–Schulz：前 8 次 `(3.4445, -4.7750, 2.0315)`，后 2 次 `(2.0, -1.5, 0.5)`。
- Standard Newton–Schulz：10 次 `(2.0, -1.5, 0.5)`。
- Muon momentum/Nesterov：严格使用 `M=mu*M+G`、`N=mu*M+G`。
- 每个逻辑矩阵在 NS 后缩放为 `sqrt(max(rows, cols)) * 0.18`。
- embedding、LM head、RMSNorm、router/vector 参数使用 AdamW。
- packed expert 3-D 参数按内部独立二维专家矩阵执行 Muon。
- 使用 native DDP；不建议直接切换到普通 DeepSpeed ZeRO。

## 不包含的内容

为便于传输，本包不包含：

- 训练数据；
- checkpoint；
- 正式训练日志；
- 模型权重；
- Python 虚拟环境和 CUDA 依赖；
- DeepSeek 官方大规模 Hybrid ZeRO/fused Muon 基础设施。

接收方需要自行准备数据，并修改三个 YAML 中的 `data.data_dir`（如目录结构不同）。

## 推荐环境

- Python 3.10+
- PyTorch，支持 CUDA/BF16/NCCL
- 8 张 GPU 时使用 `torchrun --nproc_per_node=8`
- 安装 `requirements.txt` 中依赖

## 快速检查

```bash
export PYTHONPATH=.
python -m py_compile src/muon.py src/train.py
python scripts/check_muon_config_diff.py
python tests/run_audit_tests.py
```

GPU 单步 Muon/NCCL 验证：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
PYTHONPATH=. torchrun --standalone --nproc_per_node=8 tests/gpu_nccl_muon_test.py
```


## 三组实验

```bash
bash run_three_experiments.sh
```

运行顺序：

1. `configs/rerun_adamw.yaml`
2. `configs/rerun_v4_muon_hybrid.yaml`
3. `configs/rerun_muon_standard_ns.yaml`

B 和 C 除 NS method 外保持一致。首次报告对齐比较不要提前修改 learning rate。

## 重要说明

普通 ZeRO 可能切分或展平 Muon 所需的完整逻辑矩阵。本包验证通过的是 native DDP 路径。若要使用 DeepSpeed/ZeRO，必须额外证明每个 rank 在 NS 前获得完整逻辑矩阵，并重新执行参数分组、RMS、单步和多卡一致性测试。
