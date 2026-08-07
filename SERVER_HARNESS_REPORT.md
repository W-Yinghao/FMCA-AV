# SERVER HARNESS REPORT

## 1. 服务器项目路径

`/home/infres/yinwang/FMCA-AV`

## 2. Scheduler 模式

- 当前控制 shell 位于 Slurm allocation `924528` 内，节点为 `nodecpu05`。
- 服务器策略已固定为 `mode=slurm`：无论控制 shell 是否已在 allocation 内，所有 CPU/GPU 用户任务均创建独立 `sbatch` 作业；控制 shell 只执行 `doctor/dry-run/status/stop/collect` 等管理操作。
- Slurm 命令 `sbatch`、`squeue`、`sacct`、`scancel` 可用。
- Slurm 路径固定单节点、单 task；GPU 请求使用 `--gres gpu:<1|2|4>`，并保存 Slurm job ID。
- CPU 作业固定候选分区：`CPU`。
- CIFAR、COCO 及一般 GPU 作业的 `default` profile：`V100,V100-32GB,A100,L40S,H100`。
- ImageNet 作业的 `imagenet` profile：`A100,L40S,H100`，默认排除 V100；确需 V100 回退时可由操作者显式改用 `--profile default`。

## 3. GPU 边界、型号、显存与上限

- 当前 `CUDA_VISIBLE_DEVICES`：未设置。
- 提交前具体 GPU ID：`[]`；由 Slurm 在每个作业 allocation 内分配，并由 runner 从该作业的 `CUDA_VISIBLE_DEVICES` 记录实际 ID，harness 不自行添加设备。
- 允许的 GPU 资源类型：V100/V100-32GB、A100、L40S、H100；ImageNet 默认只用 A100、L40S、H100。
- 当前控制节点没有 `nvidia-smi`，所以提交前无法可靠报告逐卡显存；GPU 作业启动后会把实际型号和显存写入对应 `environment.txt`。
- harness 硬上限：4 GPU；允许请求固定为 `0,1,2,4`。`QUEUED` 与 `RUNNING` 请求一起计入上限。

## 4. Python、PyTorch 与 CUDA

- Python：3.9.13，`/home/infres/yinwang/anaconda3/bin/python3`。
- PyTorch：未安装（`ModuleNotFoundError`）。
- PyTorch CUDA：不可用。
- 没有安装或修改 Python、CUDA、驱动及大型依赖。

## 5. FMCA-AV、FMCA、HFMCA 入口与数据目录

- 在当前项目目录内、深度最多 4 层的受限检查中，没有找到 FMCA-AV、FMCA 或 HFMCA Python 代码入口。
- 没有找到 `data/`、`datasets/`、CIFAR 或 ImageNet 数据目录。
- 当前目录原有内容是论文、实验计划、服务器指令、Codex 环境报告和 PDF；未扫描用户 home、根目录或无关目录。

## 6. 创建或修改的文件

- `harness/__init__.py`
- `harness/cli.py`
- `harness/runner.py`
- `harness/scheduler.py`
- `harness/state.py`
- `harness/config.json`
- `harness/state/jobs.json`
- `harness/state/harness.lock`
- `scripts/harness.sh`
- `runs/20260807-025006_accept-cpu-success/`
- `runs/20260807-025015_accept-cpu-fail/`
- `runs/20260807-025019_accept-cpu-fail/`
- `runs/20260807-025033_accept-stop-target/`
- `runs/20260807-025050_accept-stale-repair/`
- `runs/20260807-025212_accept-cpu-env/`
- `runs/20260807-025605_accept-slurm-cpu/`
- `runs/summary.csv`
- `SERVER_HARNESS_REPORT.md`

可直接复制的 CLI 命令：

```bash
python3 -m harness.cli doctor
python3 -m harness.cli dry-run --name <name> --gpus <0|1|2|4> --profile default -- <command> [args...]
python3 -m harness.cli submit --name <name> --gpus <0|1|2|4> --profile default -- <command> [args...]
python3 -m harness.cli dry-run --name <imagenet-name> --gpus <1|2|4> --profile imagenet -- <command> [args...]
python3 -m harness.cli submit --name <imagenet-name> --gpus <1|2|4> --profile imagenet -- <command> [args...]
python3 -m harness.cli status
python3 -m harness.cli status --run <run_id>
python3 -m harness.cli stop --run <run_id>
python3 -m harness.cli retry --run <run_id>
python3 -m harness.cli collect
python3 -m harness.cli collect --run <run_id>
```

也可使用同一入口：`scripts/harness.sh <subcommand> ...`。

## 7. 验收结果

| # | 验收项 | 结果 | 证据/说明 |
|---:|---|---|---|
| 1 | `doctor` 环境报告 | PASS | 正确报告项目、Python、PyTorch、GPU、全 Slurm 策略、分区、边界、入口及数据候选。 |
| 2 | CPU dummy：`QUEUED → RUNNING → SUCCEEDED` | PASS | `20260807-025006_accept-cpu-success`，退出码 0；另以 `20260807-025212_accept-cpu-env` 验证 CPU 命令实际收到空 `CUDA_VISIBLE_DEVICES`。 |
| 3 | 故意退出码 2 → `FAILED` | PASS | `20260807-025015_accept-cpu-fail`，退出码 2。 |
| 4 | `retry` 新目录且保留来源 | PASS | 新 run `20260807-025019_accept-cpu-fail`，`retry_from` 指向原 run，原日志未覆盖。 |
| 5 | 拒绝 5 GPU | PASS | argparse 以退出码 2 拒绝，只接受 `0,1,2,4`。 |
| 6 | 4-GPU dummy 运行时拒绝额外 1 GPU | BLOCKED | 配额逻辑检查确认 `4 active/queued + 1 requested > 4` 会立即拒绝；Slurm 4-GPU dry-run 也已通过。真实 4-GPU dummy 尚未补跑，因为当前环境没有 PyTorch/`torchrun`，而 2/4-GPU 用户命令按规则必须使用 `torchrun`；未放宽规则或安装依赖。 |
| 7 | `stop` 仅停止目标任务 | PASS | `20260807-025033_accept-stop-target` 从 `RUNNING` 变为 `STOPPED`（退出码 -15）；已完成的成功任务保持 `SUCCEEDED`。 |
| 8 | `collect` 生成汇总 | PASS | 已生成 `runs/summary.csv`；空 `metrics.jsonl` 对应的指标字段留空。 |
| 9 | 小于 30 秒的 1-GPU tensor smoke | BLOCKED | Slurm GPU 分区可选，但当前 Python 环境无 PyTorch；按指令未安装依赖。 |
| 10 | 2/4-GPU DDP dry-run | PASS | `python train.py` 正确转换为单节点 `torchrun --standalone --nnodes=1 --nproc_per_node=2 ...`；显式 4 进程命令通过转换验证；不匹配进程数和多节点命令被拒绝。Slurm dry-run 正确显示 scheduler 分配 GPU，ImageNet 4-GPU 命令只选择 A100/L40S/H100。 |

附加可靠性验收：`20260807-025050_accept-stale-repair` 的 runner 被定点 `SIGKILL` 后，下一次 `status` 自动把 stale `RUNNING` 修复为 `FAILED`；所有验收 run 当前均为终态。

全 Slurm 策略验收：CPU dummy `20260807-025605_accept-slurm-cpu` 由 `sbatch` 提交为 Slurm job `926609`，从 `QUEUED` 变为 `SUCCEEDED`，退出码 0，证明 CPU 用户命令也不在控制 shell 直接执行。

## 8. 当前阻塞

- 当前 Python 环境没有 PyTorch/CUDA。
- 当前项目目录没有 FMCA-AV/FMCA/HFMCA 可执行代码，也没有数据目录。
- 因缺少 PyTorch/`torchrun`，验收项 6 的真实 4-GPU dummy 和验收项 9 的 1-GPU tensor smoke 尚不能执行；Slurm GPU 分区本身已配置并通过 dry-run。

## 9. 下一条建议指令（未执行）

指定含 PyTorch/CUDA 的现有服务器环境，并提供 FMCA-AV/FMCA/HFMCA 代码和数据路径；随后通过 harness 的 `sbatch` 路径只补跑验收项 6、1-GPU tensor smoke 与 2/4-GPU dry-run，再更新本报告。ImageNet 使用 `--profile imagenet`，CIFAR/COCO 使用默认 profile。不要开始 Gaussian、CIFAR、COCO 或 ImageNet 正式实验。
