# QAM 复现实验说明

本仓库用于复现论文 **Q-learning with Adjoint Matching** 的主实验。

主实验入口：

```text
experiments/reproduce.py
```

完整主实验规模：

```text
16 methods × 10 domains × 5 tasks × 12 seeds = 9600 runs
```

---

## 1. 环境安装

克隆仓库：

```bash
git clone https://github.com/Arthur-Zoe/QAM-reproduce.git
cd QAM-reproduce
```

创建并激活 Conda 环境：

```bash
conda create -n qam python=3.10 -y
conda activate qam
```

安装依赖：

```bash
pip install -r requirements.txt
```

检查 JAX 是否能使用 GPU：

```bash
python3 - <<'PY'
import jax
print(jax.__version__)
print(jax.devices())
PY
```

正式复现实验前，JAX 应该能检测到 GPU / CUDA 设备。

如果只看到 CPU，说明当前 JAX 不是 GPU 版本，需要根据服务器 CUDA 版本重新安装：

```bash
pip install -U "jax[cuda12]"
# 或
pip install -U "jax[cuda13]"
```

不要在 JAX 只能检测到 CPU 的情况下提交完整实验。

---

## 2. 环境变量

运行实验前需要设置：

```bash
export MUJOCO_GL=egl
export QAM_DATA_ROOT=/path/to/ogbench_100m_data
export WANDB_PROJECT=qam-reproduce
```

如果服务器不能联网，使用 W&B 离线模式：

```bash
export WANDB_MODE=offline
```

`QAM_DATA_ROOT` 指向 OGBench 100M 数据集根目录，至少应包含：

```text
cube-quadruple-play-100m-v0/
puzzle-4x4-play-100m-v0/
```

---

## 3. 检查服务器环境

正式实验前先运行：

```bash
bash scripts/check_env.sh
```

该脚本会检查：

```text
Slurm / sbatch
GNU parallel
JAX GPU
MuJoCo
wandb
QAM_DATA_ROOT
必要数据集目录
Python 语法
```

如果缺少 `parallel`，可以安装：

```bash
conda install -c conda-forge parallel
```

只有这个检查通过后，才建议继续后续步骤。

---

## 4. 生成 Slurm 脚本

运行：

```bash
bash scripts/generate_main_sbatch.sh
```

期望输出：

```text
Formal experiment count: 9600
Debug experiment count: 160
Main sbatch generation passed.
```

如果正式实验数量不是 `9600`，不要提交实验。

注意：

```text
generate_main_sbatch.sh 会删除并重新生成 sbatch/ 目录。
不要在 sbatch/ 目录中手动保存重要脚本。
```

---

## 5. 先跑一次短 Debug

提交完整实验前，先运行一次短 debug：

```bash
bash scripts/run_debug_one.sh
```

该 debug 只跑很小的步数：

```text
offline_steps = 100
online_steps = 100
eval_episodes = 1
```

成功标准：

```text
没有 Traceback
最后出现 Local debug run finished.
```

这个 debug 只用于检查环境是否能跑通，不作为正式实验结果。

---

## 6. 提交主实验

先 dry-run，确认将要提交哪些脚本：

```bash
bash scripts/submit_main_sbatch.sh --dry-run
```

确认无误后，正式提交：

```bash
bash scripts/submit_main_sbatch.sh
```

该脚本会提交：

```text
sbatch/main-experiments-part*.sh
```

不会提交 debug 脚本。

---

## 7. 查看任务

查看当前用户的 Slurm 任务：

```bash
squeue -u $USER
```

查看历史任务记录：

```bash
sacct
```

如果使用 W&B 离线模式，实验结束后同步日志：

```bash
wandb sync wandb/offline-run-*
```

---

## 8. 推荐完整流程

```bash
conda activate qam

export MUJOCO_GL=egl
export QAM_DATA_ROOT=/path/to/ogbench_100m_data
export WANDB_PROJECT=qam-reproduce
# 如果服务器不能联网，取消下一行注释
# export WANDB_MODE=offline

bash scripts/check_env.sh
bash scripts/generate_main_sbatch.sh
bash scripts/run_debug_one.sh
bash scripts/submit_main_sbatch.sh --dry-run
bash scripts/submit_main_sbatch.sh
```

---

## 9. 当前状态

本地已完成验证：

```text
1. 可以生成主实验 sbatch 脚本
2. 正式实验数量为 9600
3. debug 实验数量为 160
4. 短 debug 可以完整跑通
5. W&B offline 模式下 run.url=None 的问题已修复
```

服务器侧仍需验证：

```text
JAX GPU
MuJoCo EGL
OGBench 数据路径
W&B 模式
Slurm / parallel
一次短 debug
```
本地 RTX 4050 已完成验证：

1. JAX GPU 可用，检测到 CudaDevice(id=0)
2. MuJoCo EGL 可用
3. W&B offline 模式可用
4. 可以生成 9600 个正式实验命令和 160 个 debug 命令
5. run_debug_one.sh 可以完整跑通，无 Traceback

服务器侧仍需验证：
1. Slurm / sbatch
2. GNU parallel
3. JAX GPU
4. OGBench 数据集路径 QAM_DATA_ROOT
5. 正式 dry-run 与提交