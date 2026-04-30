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

## 1. 推荐方式：一键初始化脚本

如果希望从 `git clone` 开始自动完成环境配置、W&B 登录、环境检查、生成 sbatch 和短 debug，可以运行：

```bash
bash scripts/bootstrap_qam_reproduce.sh
```

该脚本会依次处理：

```text
1. clone 或更新仓库
2. 创建 / 使用 Conda 环境
3. 安装 requirements.txt
4. 安装 GPU 版 JAX
5. 配置 MUJOCO_GL、QAM_DATA_ROOT、WANDB_MODE、WANDB_PROJECT
6. 引导 W&B 登录
7. 运行 W&B online smoke test
8. 运行环境检查
9. 生成主实验 sbatch 脚本
10. 运行一次短 debug
11. 执行 dry-run
12. 询问是否正式提交完整实验
```

涉及账号、数据集路径、生成 sbatch、短 debug、正式提交等步骤时，脚本会询问 `Y/N`。

注意：

```text
正式提交完整实验默认不会自动执行，需要人工确认。
```
##  当前本地验证状态

本地 RTX 4050 已完成以下验证：

```text
1. JAX GPU 可用，能检测到 CudaDevice(id=0)
2. MuJoCo EGL 可用
3. 可以生成 9600 个正式实验命令和 160 个 debug 命令
4. 短 debug 可以完整跑通
5. W&B online 模式可用，实验指标可以上传到网页端
```

服务器侧仍需验证：

```text
JAX GPU
MuJoCo EGL
OGBench 数据路径
W&B 登录与 online 上传
Slurm / parallel
一次短 debug
dry-run 提交
```
---

## 2. 手动环境安装

如果不使用一键初始化脚本，也可以手动配置环境。

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
print("JAX version:", jax.__version__)
print("JAX devices:", jax.devices())
PY
```

正式复现实验前，输出中应看到 GPU / CUDA device，不能是纯 CPU。

如果 JAX 只检测到 CPU，请根据服务器 CUDA 版本重新安装 GPU 版 JAX，例如：

```bash
pip install -U "jax[cuda12]"
# 或
pip install -U "jax[cuda13]"
```

---

## 3. 必要环境变量

运行实验前需要设置：

```bash
export MUJOCO_GL=egl
export QAM_DATA_ROOT=/path/to/ogbench_100m_data
export WANDB_MODE=online
export WANDB_PROJECT=qam-reproduce
```

如果使用 W&B 团队账号，也可以设置：

```bash
export WANDB_ENTITY=your_wandb_entity
```

`QAM_DATA_ROOT` 指向 OGBench 100M 数据集根目录，至少应包含：

```text
cube-quadruple-play-100m-v0/
puzzle-4x4-play-100m-v0/
```

例如：

```bash
export QAM_DATA_ROOT=/data/ogbench_100m_data
```

---

## 4. W&B 登录与记录

本项目使用 Weights & Biases（W&B）记录训练曲线、评估指标和实验配置。

默认假设服务器可以联网，因此正式复现实验推荐使用 online 模式。

首次使用前，在服务器上登录 W&B：

```bash
wandb login
```

终端会提示打开 W&B 网页并粘贴 API key。API key 不要发给别人，也不要提交到 GitHub。

登录成功后，可以检查：

```bash
wandb status
```

正式实验推荐设置：

```bash
export WANDB_MODE=online
export WANDB_PROJECT=qam-reproduce
```

如果使用团队项目：

```bash
export WANDB_ENTITY=your_wandb_entity
```

online 模式成功时，运行日志中应出现类似：

```text
wandb: Tracking run with wandb version ...
wandb: View project at https://wandb.ai/...
wandb: View run at https://wandb.ai/...
```

不建议在正式复现实验中使用：

```bash
export WANDB_MODE=disabled
```

因为 disabled 会关闭 W&B 记录，后续不方便汇总和对比实验结果。

如果服务器临时无法联网，可以改用 offline 模式：

```bash
export WANDB_MODE=offline
```

offline 模式下，W&B 不会上传网页端，而是把日志保存在本地。之后需要在能联网的机器上手动同步：

```bash
wandb sync wandb/offline-run-*
```

---

## 5. 检查服务器环境

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

只有该检查通过后，才建议继续后续步骤。

---

## 6. 生成 Slurm 脚本

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

## 7. 先跑一次短 Debug

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

如果使用 W&B online，运行前请确认已经设置：

```bash
export WANDB_MODE=online
export WANDB_PROJECT=qam-reproduce
```

该 debug 只用于检查环境是否能跑通，不作为正式实验结果。

---

## 8. dry-run 是什么？

正式提交前，建议先执行：

```bash
bash scripts/submit_main_sbatch.sh --dry-run
```

`dry-run` 的意思是：只打印将要提交的 sbatch 文件，但不真正提交任务。

它相当于正式提交前的演习，用于确认：

```text
1. 找到的 sbatch 文件是否正确
2. 是否只提交 main-experiments-part*.sh
3. 是否没有误提交 debug 脚本
4. 提交路径是否正确
```

对于完整主实验的 9600 runs，建议必须先 dry-run，确认无误后再正式提交。

---

## 9. 正式提交主实验

dry-run 确认无误后，正式提交：

```bash
bash scripts/submit_main_sbatch.sh
```

该脚本会提交：

```text
sbatch/main-experiments-part*.sh
```

不会提交 debug 脚本。

---

## 10. 查看任务

查看当前用户的 Slurm 任务：

```bash
squeue -u $USER
```

查看历史任务记录：

```bash
sacct
```

检查是否有失败任务：

```bash
sacct -u $USER --format=JobID,JobName,State,ExitCode,Elapsed
```

重点关注是否存在：

```text
FAILED
CANCELLED
TIMEOUT
OUT_OF_MEMORY
NODE_FAIL
```

如果存在失败任务，需要重新检查对应日志，不能直接认为完整复现成功。

---

## 11. 推荐完整手动流程

```bash
conda activate qam

wandb login

export MUJOCO_GL=egl
export QAM_DATA_ROOT=/path/to/ogbench_100m_data
export WANDB_MODE=online
export WANDB_PROJECT=qam-reproduce
# 如果使用团队账号：
# export WANDB_ENTITY=your_wandb_entity

bash scripts/check_env.sh
bash scripts/generate_main_sbatch.sh
bash scripts/run_debug_one.sh
bash scripts/submit_main_sbatch.sh --dry-run
bash scripts/submit_main_sbatch.sh
```

---

