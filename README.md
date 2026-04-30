# QAM Reproduction

This repository is for reproducing the main experiments of **Q-learning with Adjoint Matching**.

Main entry:

```text
experiments/reproduce.py
```


目标：复现 `experiments/reproduce.py` 中的主实验。

主实验规模：

```text
16 methods × 10 domains × 5 tasks × 12 seeds = 9600 runs
```

---

## 1. 环境准备

```bash
git clone https://github.com/Arthur-Zoe/QAM-reproduce.git
cd QAM-reproduce

conda create -n qam python=3.10 -y
conda activate qam
pip install -r requirements.txt
```

确认 JAX 使用 GPU：

```bash
python3 - <<'PY'
import jax
print(jax.__version__)
print(jax.devices())
PY
```

正式实验前应看到 GPU / CUDA device，不能是纯 CPU。

若需要重新安装 GPU 版 JAX，可根据服务器 CUDA 版本选择：

```bash
pip install -U "jax[cuda12]"
# or
pip install -U "jax[cuda13]"
```

---

## 2. 必要环境变量

```bash
export MUJOCO_GL=egl
export QAM_DATA_ROOT=/path/to/ogbench_100m_data
export WANDB_PROJECT=qam-reproduce
```

如果服务器不能联网：

```bash
export WANDB_MODE=offline
```

`QAM_DATA_ROOT` 下至少需要包含：

```text
cube-quadruple-play-100m-v0/
puzzle-4x4-play-100m-v0/
```

---

## 3. 服务器检查

```bash
bash scripts/check_env.sh
```

该脚本会检查：

```text
sbatch
GNU parallel
JAX GPU
MuJoCo
wandb
QAM_DATA_ROOT
必要 100M 数据目录
Python 语法
```

如果 `parallel` 缺失：

```bash
conda install -c conda-forge parallel
```

---

## 4. 生成主实验 sbatch

```bash
bash scripts/generate_main_sbatch.sh
```

期望输出：

```text
Formal experiment count: 9600
Debug experiment count: 160
Main sbatch generation passed.
```

若数量不对，不要提交正式实验。

---

## 5. 先跑一次短 debug

```bash
bash scripts/run_debug_one.sh
```

该 debug 只跑：

```text
offline_steps = 100
online_steps = 100
eval_episodes = 1
```

成功标准：

```text
无 Traceback
最后出现 Local debug run finished.
```

该步骤只用于验证服务器环境，不作为正式实验结果。

---

## 6. 正式提交主实验

先 dry-run：

```bash
bash scripts/submit_main_sbatch.sh --dry-run
```

确认提交文件无误后：

```bash
bash scripts/submit_main_sbatch.sh
```

该脚本会提交：

```text
sbatch/main-experiments-part*.sh
```

不会提交：

```text
*_debug.sh
```

---

## 7. 查看任务

```bash
squeue -u $USER
sacct
```

W&B offline 模式下，训练结束后同步：

```bash
wandb sync wandb/offline-run-*
```

---

## 8. 不建议修改的内容

除非明确排查兼容性问题，否则不建议修改：

```text
agents/
envs/
utils/
main.py
experiments/reproduce.py
experiments/generate.py
```

需要根据服务器实际情况设置的是：

```text
conda 环境
CUDA / JAX
QAM_DATA_ROOT
MUJOCO_GL
WANDB_MODE / WANDB_PROJECT / WANDB_ENTITY
Slurm / parallel
```

---

## 9. 本地已完成的验证

当前仓库已在本地完成：

```text
1. generate_main_sbatch.sh 可生成 9600 个正式主实验命令
2. debug 命令数量为 160
3. run_debug_one.sh 可完整跑通 100 offline + 100 online steps
4. 已修复 W&B offline 模式下 run.url=None 的写入问题
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