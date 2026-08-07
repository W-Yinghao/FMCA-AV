# 指令 01：在服务器建立单人、最多 4 GPU 的实验 Harness

> 将本指令原样交给服务器上的执行代理。当前指令只建立和验收 harness，不启动 FMCA-AV 正式实验。

---

## 执行指令

你正在 **服务器** 上工作。请从当前 FMCA-AV 项目目录开始，建立一个简单、可靠、单人使用的实验 harness。不要在用户本地机器上执行任何任务。

### 一、目标

完成一个 file-based 实验启动器，使后续所有 FMCA-AV 实验都通过同一入口运行，并满足：

1. 整台服务器上由本项目 harness 管理的任务，任意时刻合计最多使用 **4 张 GPU**；
2. 支持 0/1/2/4 GPU 任务；多 GPU 任务使用单节点 `torchrun`；
3. 支持 `doctor`、`dry-run`、`submit`、`status`、`stop`、`retry`、`collect`；
4. 每个任务有独立目录、标准输出/错误、状态、配置、命令和指标文件；
5. 进程退出、失败或中断后状态不会永远停留在 `RUNNING`；
6. 只有一个操作者，不需要用户系统、数据库、Web UI、Redis、W&B、Git、commit、SHA 或任何文件 hash。

### 二、边界条件

- 不启动正式训练；只允许 CPU dummy、1-GPU 极小 smoke，以及必要的 2/4-GPU `dry-run`。
- 不下载数据集，不安装 CUDA/驱动，不修改系统 Python，不申请超过 4 张 GPU。
- 不删除或覆盖现有代码、数据、checkpoint 和日志。
- 不递归扫描 `/`、用户 home 或无关目录；只检查当前项目目录、明确的数据目录和当前 scheduler/GPU 环境。
- 若服务器使用 Slurm：
  - 在 login node 上不得直接占用 GPU；
  - 通过 `sbatch` 提交，单任务 `--gres` 不超过 4；
  - 若已经处于 Slurm allocation 内，只使用 `CUDA_VISIBLE_DEVICES` 暴露的卡。
- 若没有 scheduler：使用 direct subprocess 模式，并由 harness 自己分配允许的 GPU ID。
- 若 `CUDA_VISIBLE_DEVICES` 已设置，将它视为绝对边界，不得添加其中没有的 GPU。
- 若未设置 `CUDA_VISIBLE_DEVICES`，从 `nvidia-smi` 读取 GPU，并最多把前 4 张写入配置；在报告中明确列出。
- 若没有 PyTorch 或 GPU，不要自行安装大型依赖；完成 CPU harness 验收并报告阻塞。

### 三、保持实现简单

使用 Python 标准库实现 file-based harness。除项目已经存在且稳定的依赖外，不为 harness 引入第三方包。

创建以下结构；若同名目录已存在，先检查并兼容，不覆盖用户内容：

```text
harness/
  __init__.py
  cli.py
  runner.py
  scheduler.py
  state.py
  config.json
  state/
    jobs.json
    harness.lock
runs/
scripts/
  harness.sh
SERVER_HARNESS_REPORT.md
```

`harness/config.json` 至少包含：

```json
{
  "max_gpus": 4,
  "allowed_gpu_ids": [],
  "mode": "auto",
  "runs_dir": "runs",
  "stop_grace_seconds": 30,
  "poll_seconds": 5
}
```

要求：

- `max_gpus` 必须硬限制为 4；命令行不得把它提高到 5 或更多；
- `allowed_gpu_ids` 由当前 allocation / `CUDA_VISIBLE_DEVICES` / `nvidia-smi` 初始化；
- `mode=auto` 只允许解析为 `slurm` 或 `direct`；
- 使用 Linux 文件锁，防止两个 submit 同时绕过 4-GPU 上限；
- `jobs.json` 使用临时文件 + 原子替换写入，避免中断后 JSON 损坏；
- 不使用 `shell=True` 拼接训练命令；命令以参数列表传给 subprocess；
- direct 模式使用独立 process group；`stop` 先发 `SIGTERM`，等待 grace period 后才发 `SIGKILL`；
- Slurm 模式保存 scheduler job ID，用 `squeue/sacct/scancel` 更新状态；
- 每次 `status` 都检查真实 PID/Slurm 状态，并修复 stale `RUNNING`。

### 四、任务目录规范

任务 ID 使用人可读格式，不使用 hash：

```text
YYYYMMDD-HHMMSS_<experiment-name>
```

每个 `runs/<run_id>/` 至少包含：

```text
command.txt
request.json
environment.txt
stdout.log
stderr.log
status.json
metrics.jsonl
checkpoints/
```

`status.json` 至少记录：

- `run_id`；
- `name`；
- `state`: `QUEUED/RUNNING/SUCCEEDED/FAILED/STOPPED/BLOCKED`；
- `requested_gpus` 与实际 GPU IDs；
- PID 或 Slurm job ID；
- start/end time；
- exit code；
- failure reason；
- retry 来源（如有）。

不记录 Git/SHA/hash。环境信息只需保存：hostname、Python、PyTorch/CUDA（若存在）、GPU 型号与显存、scheduler、当前项目路径。

### 五、CLI 行为

实现以下入口，具体 argparse 语法可以微调，但报告中必须给出可直接复制的命令：

```bash
python -m harness.cli doctor
python -m harness.cli dry-run --name <name> --gpus <0|1|2|4> -- <command> [args...]
python -m harness.cli submit  --name <name> --gpus <0|1|2|4> -- <command> [args...]
python -m harness.cli status [--run <run_id>]
python -m harness.cli stop --run <run_id>
python -m harness.cli retry --run <run_id>
python -m harness.cli collect [--run <run_id>]
```

行为要求：

- `doctor`：只读报告项目目录、Python、PyTorch、CUDA、GPU、scheduler、可用 GPU 边界、代码入口和数据目录候选；
- `dry-run`：解析最终命令、环境和 GPU 分配，但不启动进程；
- `submit`：加锁，计算当前 harness 已占 GPU 数；若新任务导致总数超过 4，立即拒绝并给出非零退出码；
- `status`：显示所有任务的 run ID、状态、GPU、时长和退出码；
- `stop`：只停止指定任务，不影响其他任务；
- `retry`：新建一个可读的新 run ID，保留原任务引用；不得覆盖原日志；
- `collect`：把已完成任务的 `status.json` 和 `metrics.jsonl` 汇总成 `runs/summary.csv`；缺失指标时留空，不报假成功。

对请求 `--gpus 3` 或 `--gpus 5` 必须拒绝；当前允许集合固定为 `0,1,2,4`。

### 六、DDP 命令规则

- 0 GPU：设置空的 `CUDA_VISIBLE_DEVICES`，确保 CPU 任务不会误占 GPU；
- 1 GPU：原样启动用户命令，只暴露 harness 分配的一张卡；
- 2/4 GPU：
  - 若命令已经以 `torchrun` 开头，验证 `--nproc_per_node` 与请求卡数一致；
  - 若命令形如 `python train.py ...`，去掉前导 Python executable 后生成：

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=<2-or-4> train.py ...
```

- 其他命令格式不得猜测转换，直接拒绝并提示操作者给出显式 `torchrun` 命令；
- 不支持多节点；
- 设置 `OMP_NUM_THREADS` 为保守值，并允许在 config 中调整；
- 不擅自改变训练 batch size、学习率或 gradient accumulation；这些属于后续实验配置。

### 七、验收测试

依次执行并记录结果：

1. `doctor` 成功生成服务器环境报告；
2. CPU dummy job 正常从 `QUEUED → RUNNING → SUCCEEDED`；
3. 一个故意退出码为 2 的 CPU job 正确变成 `FAILED`；
4. `retry` 生成新目录且不覆盖原任务；
5. 请求 5 GPU 被拒绝；
6. 提交一个请求 4 GPU、仅执行 `sleep 10` 的短 dummy，使其处于 `RUNNING` 后再提交 1-GPU 任务；后者必须被 harness 拒绝，dummy 结束后卡应自动释放；
7. `stop` 只终止目标 dummy job；
8. `collect` 生成 `runs/summary.csv`；
9. 如果 PyTorch 和 GPU 已就绪，再运行一个小于 30 秒的 1-GPU tensor smoke；否则标记 `BLOCKED`，不要安装依赖；
10. 2/4-GPU 只验证 `dry-run` 生成的 `torchrun` 命令和 GPU 边界，不做正式 DDP 训练。

### 八、完成后输出

更新 `SERVER_HARNESS_REPORT.md`，只报告：

1. 服务器项目路径；
2. scheduler 模式；
3. 允许的 GPU ID、型号、显存与 4-GPU 上限；
4. Python/PyTorch/CUDA 状态；
5. 找到的 FMCA-AV/FMCA/HFMCA 代码入口和数据目录；
6. 创建/修改的文件；
7. 每项验收测试的 `PASS/FAIL/BLOCKED`；
8. 当前阻塞；
9. 下一条建议指令，但不要自行执行。

完成 harness 和上述验收后停止。不要开始 Gaussian、CIFAR 或 ImageNet 实验。
