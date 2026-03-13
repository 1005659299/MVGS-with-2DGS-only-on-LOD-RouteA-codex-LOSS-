# AutoDL 镜像选择与环境配置指南

> 本指南面向在 [AutoDL](https://www.autodl.com) 平台上租用 GPU 训练
> **MVGS-with-2DGS-only-on-LOD-RouteA-codex-LOSS** 模型的用户。
> 重点说明如何选择正确的「基础镜像」，以及创建实例后如何完成环境搭建。

---

## 目录

1. [镜像选择总览](#1-镜像选择总览)
2. [按 GPU 型号选择镜像](#2-按-gpu-型号选择镜像)
3. [创建实例后的完整安装步骤](#3-创建实例后的完整安装步骤)
4. [验证环境](#4-验证环境)
5. [常见问题排查](#5-常见问题排查)

---

## 1. 镜像选择总览

### 1.1 项目软件依赖（来自 `environment.yml`）

| 组件 | 原始要求 | 说明 |
|---|---|---|
| Python | 3.7.13 | 较老，建议升级到 3.8+ |
| PyTorch | 1.12.1 | 原始版本 |
| CUDA Toolkit | 11.6 | 原始版本 |
| torchvision | 0.13.1 | |
| torchaudio | 0.12.1 | |
| 其他 | plyfile, tqdm | pip 安装 |

### 1.2 关键约束

本项目包含 **3 个自定义 CUDA C++ 扩展**，必须在实例上从源码编译：

- `submodules/diff-surfel-rasterization` — 2D-GS surfel 光栅化
- `submodules/diff-gaussian-rasterization` — 3D-GS EWA 光栅化
- `submodules/simple-knn` — KNN 距离计算

编译这些扩展需要：
- **NVIDIA GPU**（必须，不支持华为昇腾/摩尔线程）
- **CUDA Toolkit**（nvcc 编译器）
- **PyTorch** 对应 CUDA 版本的安装
- **C++ 编译器**（gcc/g++）

### 1.3 AutoDL 镜像类型说明

在 AutoDL 创建实例页面，「镜像」区域有以下选项：

| 镜像类型 | 说明 | 推荐度 |
|---|---|---|
| **基础镜像** | 包含深度学习框架（PyTorch/TensorFlow）+ Miniconda + CUDA | ✅ **推荐** |
| 社区镜像 | 其他用户分享的预配置环境 | ⚠️ 看具体内容 |
| 自定义镜像 | 你自己保存的镜像 | 如有则用 |

**结论：选择「基础镜像」是正确的。** 接下来关键是选对 **PyTorch 版本 + CUDA 版本** 的组合。

---

## 2. 按 GPU 型号选择镜像

### ⚠️ 重要原则

**不同 GPU 架构需要不同的最低 CUDA 版本。** 如果 CUDA 版本太低，GPU 将无法被识别。

### 2.1 镜像选择对照表

| 租用的 GPU | GPU 架构 | 最低 CUDA | **推荐 AutoDL 基础镜像** |
|---|---|---|---|
| **RTX 3090** | Ampere | 11.1 | `PyTorch 1.12.1` + `CUDA 11.6` + `Python 3.8` |
| **RTX 4090** | Ada Lovelace | 11.8 | `PyTorch 2.1.0` + `CUDA 12.1` + `Python 3.10` |
| **RTX 5090** | Blackwell | 12.8 | `PyTorch 2.6.0` + `CUDA 12.6` + `Python 3.11`（或平台可用最新版） |
| **L20** | Ada Lovelace | 12.0 | `PyTorch 2.1.0` + `CUDA 12.1` + `Python 3.10` |
| **V100** | Volta | 10.0 | `PyTorch 1.12.1` + `CUDA 11.6` + `Python 3.8` |
| **A800-80GB** | Ampere | 11.1 | `PyTorch 1.12.1` + `CUDA 11.6` + `Python 3.8` |
| **H800** | Hopper | 11.8 | `PyTorch 2.1.0` + `CUDA 12.1` + `Python 3.10` |
| **H20** | Hopper | 11.8 | `PyTorch 2.1.0` + `CUDA 12.1` + `Python 3.10` |
| **PRO 6000** | Ada Lovelace | 12.0 | `PyTorch 2.1.0` + `CUDA 12.1` + `Python 3.10` |

> **说明：** AutoDL 基础镜像的可选版本可能随时更新。
> 请选择平台上**与推荐最接近**的版本。优先保证 CUDA 版本 ≥ 上表中的最低要求。

### 2.2 最常见的两种情况

#### 情况 A：租用 RTX 3090 / V100 / A800（老架构，CUDA 11.6 兼容）

```
推荐镜像：PyTorch 1.12.1 / CUDA 11.6 / Python 3.8
```

这与项目 `environment.yml` 原始配置完全匹配，安装最简单。

#### 情况 B：租用 RTX 4090 / L20 / H800 / H20 / PRO 6000 / RTX 5090（新架构）

```
推荐镜像：PyTorch 2.1.0 / CUDA 12.1 / Python 3.10
（或平台上可用的 PyTorch 2.x + CUDA 12.x 最新版）
```

新架构 GPU 不兼容 CUDA 11.6，必须使用 CUDA 11.8 或更高版本。
代码本身兼容 PyTorch 2.x，但自定义 CUDA 扩展需要重新编译（安装步骤中会处理）。

---

## 3. 创建实例后的完整安装步骤

### 3.1 通用步骤（所有 GPU）

创建实例并进入终端后，执行以下命令：

```bash
# ============ 步骤 1：检查 GPU 是否可用 ============
nvidia-smi
# 应显示你租用的 GPU 型号和 CUDA 版本

# ============ 步骤 2：克隆仓库 ============
cd /root/autodl-tmp  # AutoDL 数据盘，空间更大
git clone https://github.com/1005659299/MVGS-with-2DGS-only-on-LOD-RouteA-codex-LOSS-.git --recursive
cd MVGS-with-2DGS-only-on-LOD-RouteA-codex-LOSS-

# 确认子模块已下载
ls submodules/
# 应看到：diff-gaussian-rasterization  diff-surfel-rasterization  simple-knn
# 如果目录为空，运行：
git submodule update --init --recursive

# ============ 步骤 3：检查 Python 和 PyTorch ============
python --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### 3.2 情况 A 安装（CUDA 11.6 镜像）

如果你选择了 `PyTorch 1.12.1 + CUDA 11.6` 镜像（适用于 RTX 3090 / V100 / A800）：

```bash
# PyTorch 已在镜像中预装，直接安装其他依赖
pip install plyfile tqdm tensorboard

# 编译安装自定义 CUDA 扩展
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn

# 安装 diff-surfel-rasterization（2D-GS 核心组件）
pip install submodules/diff-surfel-rasterization

# 安装 lpips（用于评估指标）
pip install lpips
```

### 3.3 情况 B 安装（CUDA 12.x 镜像）

如果你选择了 `PyTorch 2.x + CUDA 12.x` 镜像（适用于 RTX 4090 / L20 / H800 / RTX 5090 等）：

```bash
# PyTorch 2.x 已在镜像中预装，直接安装其他依赖
pip install plyfile tqdm tensorboard

# 编译安装自定义 CUDA 扩展（与情况 A 命令相同，但会用 CUDA 12.x 编译）
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
pip install submodules/diff-surfel-rasterization

# 安装 lpips
pip install lpips
```

> **注意：** 如果编译报错，可能需要安装 ninja 加速编译：
> ```bash
> pip install ninja
> ```

### 3.4 验证 CUDA 扩展编译成功

```bash
python -c "
import diff_gaussian_rasterization
print('diff_gaussian_rasterization OK')

import diff_surfel_rasterization
print('diff_surfel_rasterization OK')

from simple_knn._C import distCUDA2
print('simple_knn OK')

print('所有 CUDA 扩展编译成功！')
"
```

### 3.5 下载数据集

```bash
# 示例：下载 Mip-NeRF360 数据集
cd /root/autodl-tmp
mkdir -p data && cd data

# 方式 1：AutoDL 数据盘上传（推荐，避免外网下载慢）
# 在 AutoDL 控制台 → 文件上传 → 上传数据集压缩包

# 方式 2：wget（如果网络可用）
# wget https://storage.googleapis.com/gresearch/refraw360/360_v2.zip
# unzip 360_v2.zip
```

### 3.6 开始训练

```bash
cd /root/autodl-tmp/MVGS-with-2DGS-only-on-LOD-RouteA-codex-LOSS-

# ---- 根据 GPU 显存选择合适命令 ----

# 24GB（RTX 4090 / RTX 3090）→ 降低 MV
python train.py -s /root/autodl-tmp/data/360_v2/garden \
    --eval --white_background \
    -m /root/autodl-tmp/output/garden \
    --mv 4

# 32GB（RTX 5090 / V100）→ 中等 MV
python train.py -s /root/autodl-tmp/data/360_v2/garden \
    --eval --white_background \
    -m /root/autodl-tmp/output/garden \
    --mv 6

# 48GB+（L20 / PRO 6000 / H800 / H20 / A800）→ 完整训练
python run_360.py
```

---

## 4. 验证环境

训练启动后，应看到类似输出：

```
loading ckpt ...
resolution: -1 MV: 4
Training progress:   0%|          | 0/30000 ...
```

如果能正常开始迭代且无报错，说明环境配置成功。

### 监控 GPU 使用

```bash
# 另一个终端窗口
watch -n 1 nvidia-smi
```

应看到 GPU 利用率 > 80%，显存使用接近你选的 `--mv` 值对应的预期。

---

## 5. 常见问题排查

### Q1: 编译 CUDA 扩展时报错 `nvcc fatal: Unsupported gpu architecture`

**原因：** CUDA 版本不支持你的 GPU 架构。

**解决：** 确认你选择的镜像 CUDA 版本符合 [第 2 节对照表](#21-镜像选择对照表)。
例如 RTX 4090 至少需要 CUDA 11.8，建议 CUDA 12.1+。

```bash
# 检查当前 CUDA 版本
nvcc --version
```

### Q2: `import diff_surfel_rasterization` 报 `ModuleNotFoundError`

**原因：** CUDA 扩展未成功编译安装。

**解决：**
```bash
# 重新编译，并查看详细错误信息
cd submodules/diff-surfel-rasterization
pip install -e . --verbose 2>&1 | tee build.log
# 查看 build.log 中的错误
```

### Q3: `RuntimeError: CUDA out of memory`

**原因：** 显存不足。

**解决：** 降低 `--mv` 值：
```bash
# 从 mv 6 降到 mv 4 或 mv 2
python train.py -s {data_path} --eval --white_background -m {save_path} --mv 2
```

### Q4: `torch.cuda.is_available()` 返回 `False`

**原因：** PyTorch 的 CUDA 版本与系统不匹配。

**解决：**
```bash
# 检查系统 CUDA
nvidia-smi  # 右上角显示驱动支持的 CUDA 版本
nvcc --version  # 编译器 CUDA 版本

# 重装匹配版本的 PyTorch
# 例如 CUDA 12.1：
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 --index-url https://download.pytorch.org/whl/cu121
```

### Q5: 子模块目录为空（submodules/ 下无文件）

**原因：** 克隆时未加 `--recursive`。

**解决：**
```bash
git submodule update --init --recursive
```

### Q6: 编译时 `g++` 版本不兼容

**解决：**
```bash
# 安装兼容的 g++ 版本
apt update && apt install -y g++-9
export CC=gcc-9
export CXX=g++-9
# 重新编译
pip install submodules/diff-gaussian-rasterization --force-reinstall --no-cache-dir
```

---

## 附录：AutoDL 镜像选择速查表

| 你租的 GPU | 选这个镜像 | 训练命令关键参数 |
|---|---|---|
| RTX 3090（24GB） | `PyTorch 1.12.1 / CUDA 11.6 / Python 3.8` | `--mv 4` |
| RTX 4090（24GB） | `PyTorch 2.1.0 / CUDA 12.1 / Python 3.10` | `--mv 4` |
| RTX 5090（32GB） | `PyTorch 2.x / CUDA 12.6+ / Python 3.11` | `--mv 6` |
| V100（32GB） | `PyTorch 1.12.1 / CUDA 11.6 / Python 3.8` | `--mv 6` |
| L20（48GB） | `PyTorch 2.1.0 / CUDA 12.1 / Python 3.10` | `run_360.py` |
| A800-80GB | `PyTorch 1.12.1 / CUDA 11.6 / Python 3.8` | `run_360.py` |
| H800（80GB） | `PyTorch 2.1.0 / CUDA 12.1 / Python 3.10` | `run_360.py` |
| H20（96GB） | `PyTorch 2.1.0 / CUDA 12.1 / Python 3.10` | `run_360.py` |
| PRO 6000（96GB） | `PyTorch 2.1.0 / CUDA 12.1 / Python 3.10` | `run_360.py` |
