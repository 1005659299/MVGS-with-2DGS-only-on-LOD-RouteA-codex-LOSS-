# Open Interpreter + AutoDL 远程训练完整指南

> **场景：** 你在 Windows 系统上，通过 Ubuntu (WSL) 中安装的 Open Interpreter，
> 连接 AutoDL 租用的 GPU 实例，上传本地代码与数据集，完成 MVGS 模型训练。

---

## 目录

1. [方案总览与对比](#1-方案总览与对比)
2. [前提准备](#2-前提准备)
3. [方案 A：Open Interpreter 全自动化（推荐）](#3-方案-a-open-interpreter-全自动化推荐)
4. [方案 B：本地脚本半自动化](#4-方案-b-本地脚本半自动化)
5. [方案 C：纯手动 SSH 操作](#5-方案-c-纯手动-ssh-操作)
6. [训练结果回传与查看](#6-训练结果回传与查看)
7. [安全提醒](#7-安全提醒)
8. [常见问题](#8-常见问题)

---

## 1. 方案总览与对比

| 方案 | 自动化程度 | 难度 | 适合场景 |
|---|---|---|---|
| **A：Open Interpreter** | ⭐⭐⭐⭐⭐ | 中 | 想用自然语言指挥 AI 自动完成所有步骤 |
| **B：本地脚本** | ⭐⭐⭐ | 低 | 只想一键运行，不需要 AI 交互 |
| **C：纯手动** | ⭐ | 低 | 学习目的，理解每一步操作 |

**核心流程（三个方案共通）：**

```
本地 Windows/WSL                        AutoDL 远程 GPU 实例
┌─────────────────┐    SSH/SCP     ┌──────────────────────┐
│  代码仓库        │ ──────────→  │  /root/autodl-tmp/    │
│  数据集          │              │  ├── code/            │
│  Open Interpreter│ ←──────────  │  ├── data/            │
│                  │   结果回传    │  └── output/          │
└─────────────────┘              └──────────────────────┘
```

---

## 2. 前提准备

### 2.1 AutoDL 实例（已完成）

1. 在 AutoDL 创建实例（参考 [AUTODL_SETUP_GUIDE.md](AUTODL_SETUP_GUIDE.md) 选择镜像）
2. 实例创建后，在控制台获取 **SSH 登录信息**，格式如下：

   ```
   SSH 指令: ssh -p 12345 root@connect.bjb1.autodl.com
   密码:     xxxxxxxx
   ```

   记下这三个值：
   - **主机名 (HOST)**: `connect.bjb1.autodl.com`
   - **端口 (PORT)**: `12345`（示例，以实际为准）
   - **密码 (PASSWORD)**: 控制台显示的密码

### 2.2 Windows + WSL 环境

打开 Windows 终端（PowerShell），确认 WSL 已安装：

```powershell
wsl --list --verbose
```

进入 Ubuntu (WSL)：

```powershell
wsl
```

### 2.3 在 WSL Ubuntu 中安装工具

```bash
# 确保 ssh 和 scp 可用（通常 WSL 自带）
which ssh scp
# 输出应为 /usr/bin/ssh 和 /usr/bin/scp

# 安装 sshpass（用于脚本中自动输入密码）
sudo apt update && sudo apt install -y sshpass

# 安装 rsync（更高效的文件传输）
sudo apt install -y rsync
```

### 2.4 安装 Open Interpreter

```bash
# 确保 Python 3.10+ 已安装
python3 --version

# 安装 Open Interpreter
pip install open-interpreter

# 验证
interpreter --version
```

### 2.5 准备本地文件

确认以下文件在 WSL 中可访问：

```bash
# 代码仓库（Windows 磁盘在 WSL 中挂载为 /mnt/c/ 等）
ls /mnt/c/Users/你的用户名/path/to/MVGS-with-2DGS-only-on-LOD-RouteA-codex-LOSS-/

# 数据集（例如 Mip-NeRF360 数据）
ls /mnt/c/Users/你的用户名/path/to/data/360_v2/
```

> **提示：** 建议先将文件复制到 WSL 本地路径（如 `~/mvgs/`），
> 从 `/mnt/c/` 传输会慢很多。
>
> ```bash
> mkdir -p ~/mvgs
> cp -r /mnt/c/Users/你的用户名/path/to/MVGS-with-2DGS-only-on-LOD-RouteA-codex-LOSS- ~/mvgs/code
> cp -r /mnt/c/Users/你的用户名/path/to/data ~/mvgs/data
> ```

---

## 3. 方案 A：Open Interpreter 全自动化（推荐）

### 3.1 工作原理

Open Interpreter 可以在本地执行 shell 命令。你用**自然语言**告诉它要做什么，
它会自动生成并执行 SSH/SCP 命令来操作远程 AutoDL 实例。

### 3.2 启动 Open Interpreter

```bash
# 在 WSL 终端中启动
interpreter
```

### 3.3 给 Open Interpreter 的提示模板

将以下内容复制粘贴给 Open Interpreter（替换 `<>` 中的实际值）：

---

#### 第一步：配置 SSH 连接信息

```
我需要你帮我在 AutoDL 远程 GPU 服务器上训练一个模型。以下是连接信息：

- SSH 主机: <connect.bjb1.autodl.com>
- SSH 端口: <12345>
- 用户名: root
- 密码: <你的密码>

请先用 sshpass 测试 SSH 连接是否正常，运行 nvidia-smi 确认 GPU 可用。
```

> Open Interpreter 会执行类似以下命令：
> ```bash
> sshpass -p '你的密码' ssh -p 12345 -o StrictHostKeyChecking=no root@connect.bjb1.autodl.com 'nvidia-smi'
> ```

---

#### 第二步：上传代码和数据

```
现在请帮我把本地文件上传到远程服务器：

1. 代码仓库在: ~/mvgs/code/
   上传到远程: /root/autodl-tmp/MVGS/

2. 数据集在: ~/mvgs/data/
   上传到远程: /root/autodl-tmp/data/

请用 rsync 或 scp 上传，显示进度。
```

> Open Interpreter 会执行类似：
> ```bash
> sshpass -p '密码' rsync -avz --progress -e 'ssh -p 12345' \
>   ~/mvgs/code/ root@connect.bjb1.autodl.com:/root/autodl-tmp/MVGS/
>
> sshpass -p '密码' rsync -avz --progress -e 'ssh -p 12345' \
>   ~/mvgs/data/ root@connect.bjb1.autodl.com:/root/autodl-tmp/data/
> ```

---

#### 第三步：远程安装依赖

```
代码和数据已上传。现在请在远程服务器上安装训练所需的依赖。

在远程执行以下步骤：
1. cd /root/autodl-tmp/MVGS
2. pip install plyfile tqdm tensorboard lpips ninja
3. pip install submodules/diff-gaussian-rasterization
4. pip install submodules/simple-knn
5. pip install submodules/diff-surfel-rasterization
6. 安装完成后，运行验证命令确认 CUDA 扩展正常：
   python -c "import diff_gaussian_rasterization; import diff_surfel_rasterization; from simple_knn._C import distCUDA2; print('All OK')"
```

---

#### 第四步：开始训练

根据你租用的 GPU 显存选择合适的命令：

**24GB GPU（RTX 4090 / RTX 3090）：**
```
请在远程服务器上开始训练：

cd /root/autodl-tmp/MVGS
nohup python train.py \
  -s /root/autodl-tmp/data/360_v2/garden \
  --eval --white_background \
  -m /root/autodl-tmp/output/garden \
  --mv 4 \
  > /root/autodl-tmp/train.log 2>&1 &

然后查看训练日志前 20 行确认训练已开始。
```

**48GB+ GPU（L20 / H800 / H20 等）：**
```
请在远程服务器上开始完整训练：

cd /root/autodl-tmp/MVGS
nohup python run_360.py \
  --data_root /root/autodl-tmp/data \
  --output_dir /root/autodl-tmp/output \
  > /root/autodl-tmp/train.log 2>&1 &

然后查看训练日志确认训练已开始。
```

> **关键：使用 `nohup ... &`** 让训练在后台运行。
> 这样即使你断开 SSH 连接，训练也会继续。

---

#### 第五步：监控训练

```
请帮我查看远程服务器上的训练进度：
1. 查看 GPU 使用情况：nvidia-smi
2. 查看训练日志最后 30 行：tail -30 /root/autodl-tmp/train.log
3. 检查训练进程是否还在运行：ps aux | grep train.py
```

---

#### 第六步：下载训练结果

```
训练完成后，请帮我把结果下载到本地：

远程结果目录: /root/autodl-tmp/output/garden/
下载到本地: ~/mvgs/results/

请用 rsync 下载，显示进度。
```

### 3.4 完整的一次性指令（高级用户）

你也可以将所有步骤合并成一条指令给 Open Interpreter：

```
我需要你帮我完成以下远程 GPU 训练任务：

SSH 信息：
- 主机: connect.bjb1.autodl.com，端口: 12345，用户: root，密码: xxxxxxxx

请按顺序执行：
1. 测试 SSH 连接，运行 nvidia-smi 确认 GPU
2. 用 rsync 上传 ~/mvgs/code/ 到远程 /root/autodl-tmp/MVGS/
3. 用 rsync 上传 ~/mvgs/data/ 到远程 /root/autodl-tmp/data/
4. 在远程安装依赖：pip install plyfile tqdm tensorboard lpips ninja
5. 在远程编译 CUDA 扩展：
   cd /root/autodl-tmp/MVGS && pip install submodules/diff-gaussian-rasterization submodules/simple-knn submodules/diff-surfel-rasterization
6. 在远程启动训练（后台运行）：
   cd /root/autodl-tmp/MVGS && nohup python train.py -s /root/autodl-tmp/data/360_v2/garden --eval --white_background -m /root/autodl-tmp/output/garden --mv 4 > /root/autodl-tmp/train.log 2>&1 &
7. 等待 10 秒后查看日志确认训练已开始

每一步执行前请先告诉我你要做什么，等我确认。
```

---

## 4. 方案 B：本地脚本半自动化

如果你更喜欢一键运行脚本而非 AI 对话，可以使用以下 Bash 脚本。

### 4.1 创建自动化脚本

在 WSL 中创建文件 `~/mvgs/autodl_train.sh`：

```bash
#!/bin/bash
# ============================================================
# AutoDL 远程训练自动化脚本
# 用法: bash autodl_train.sh
# ============================================================

# ---- 配置区（修改为你的实际信息） ----
AUTODL_HOST="connect.bjb1.autodl.com"
AUTODL_PORT="12345"                # AutoDL SSH 端口
AUTODL_PASS="${AUTODL_PASS:-}"       # 通过环境变量传入，避免硬编码
# 用法: AUTODL_PASS="你的密码" bash autodl_train.sh
if [ -z "$AUTODL_PASS" ]; then
    read -sp "请输入 AutoDL SSH 密码: " AUTODL_PASS
    echo
fi

LOCAL_CODE="$HOME/mvgs/code"       # 本地代码目录
LOCAL_DATA="$HOME/mvgs/data"       # 本地数据集目录
LOCAL_RESULTS="$HOME/mvgs/results" # 本地结果保存目录

REMOTE_BASE="/root/autodl-tmp"
REMOTE_CODE="$REMOTE_BASE/MVGS"
REMOTE_DATA="$REMOTE_BASE/data"
REMOTE_OUTPUT="$REMOTE_BASE/output"

MV_VALUE=4                         # 多视图数（按显存调整：24GB→4, 32GB→6, 48GB+→8）
SCENE="garden"                     # 训练场景名称
DATASET_SUBDIR="360_v2"            # 数据集子目录
# ---- 配置区结束 ----

SSH_CMD="sshpass -p '$AUTODL_PASS' ssh -p $AUTODL_PORT -o StrictHostKeyChecking=accept-new root@$AUTODL_HOST"
SCP_CMD="sshpass -p '$AUTODL_PASS' rsync -avz --progress -e 'ssh -p $AUTODL_PORT -o StrictHostKeyChecking=accept-new'"

echo "============================================"
echo "  AutoDL 远程训练自动化脚本"
echo "============================================"

# 步骤 1：测试连接
echo ""
echo "[步骤 1/6] 测试 SSH 连接..."
eval $SSH_CMD 'nvidia-smi'
if [ $? -ne 0 ]; then
    echo "❌ SSH 连接失败！请检查主机、端口和密码。"
    exit 1
fi
echo "✅ SSH 连接成功"

# 步骤 2：上传代码
echo ""
echo "[步骤 2/6] 上传代码到远程..."
eval $SCP_CMD "$LOCAL_CODE/" "root@$AUTODL_HOST:$REMOTE_CODE/"
echo "✅ 代码上传完成"

# 步骤 3：上传数据
echo ""
echo "[步骤 3/6] 上传数据到远程..."
eval $SCP_CMD "$LOCAL_DATA/" "root@$AUTODL_HOST:$REMOTE_DATA/"
echo "✅ 数据上传完成"

# 步骤 4：安装依赖
echo ""
echo "[步骤 4/6] 安装依赖..."
eval $SSH_CMD "cd $REMOTE_CODE && pip install plyfile tqdm tensorboard lpips ninja && \
    pip install submodules/diff-gaussian-rasterization && \
    pip install submodules/simple-knn && \
    pip install submodules/diff-surfel-rasterization"
echo "✅ 依赖安装完成"

# 步骤 5：启动训练
echo ""
echo "[步骤 5/6] 启动训练（后台运行）..."
eval $SSH_CMD "cd $REMOTE_CODE && \
    nohup python train.py \
        -s $REMOTE_DATA/$DATASET_SUBDIR/$SCENE \
        --eval --white_background \
        -m $REMOTE_OUTPUT/$SCENE \
        --mv $MV_VALUE \
        > $REMOTE_BASE/train.log 2>&1 &"
echo "✅ 训练已在后台启动"

# 步骤 6：确认
echo ""
echo "[步骤 6/6] 检查训练状态..."
sleep 5
eval $SSH_CMD "tail -10 $REMOTE_BASE/train.log"

echo ""
echo "============================================"
echo "  训练已启动！"
echo ""
echo "  监控训练:"
echo "    $SSH_CMD 'tail -f $REMOTE_BASE/train.log'"
echo ""
echo "  下载结果（训练完成后）:"
echo "    mkdir -p $LOCAL_RESULTS"
echo "    $SCP_CMD root@$AUTODL_HOST:$REMOTE_OUTPUT/$SCENE/ $LOCAL_RESULTS/$SCENE/"
echo "============================================"
```

### 4.2 运行

```bash
chmod +x ~/mvgs/autodl_train.sh
bash ~/mvgs/autodl_train.sh
```

---

## 5. 方案 C：纯手动 SSH 操作

如果你想手动操作每一步：

### 5.1 从 WSL 连接到 AutoDL

```bash
# 直接 SSH 登录（会提示输入密码）
ssh -p 12345 root@connect.bjb1.autodl.com
```

### 5.2 上传文件（另开一个 WSL 终端）

```bash
# 上传代码
scp -r -P 12345 ~/mvgs/code/ root@connect.bjb1.autodl.com:/root/autodl-tmp/MVGS/

# 上传数据集
scp -r -P 12345 ~/mvgs/data/ root@connect.bjb1.autodl.com:/root/autodl-tmp/data/
```

> **大数据集提示：** `scp` 不支持断点续传。如果数据集很大（>10GB），
> 建议使用 `rsync`：
> ```bash
> rsync -avz --progress -e 'ssh -p 12345' ~/mvgs/data/ root@connect.bjb1.autodl.com:/root/autodl-tmp/data/
> ```

### 5.3 在远程安装依赖并训练

```bash
# 在 SSH 连接的远程终端中执行
cd /root/autodl-tmp/MVGS

pip install plyfile tqdm tensorboard lpips ninja
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
pip install submodules/diff-surfel-rasterization

# 启动训练（使用 tmux 或 nohup 防止断连中断）
tmux new -s train
python train.py -s /root/autodl-tmp/data/360_v2/garden \
    --eval --white_background \
    -m /root/autodl-tmp/output/garden \
    --mv 4
# Ctrl+B, D 分离 tmux 会话
# tmux attach -t train 重新连接
```

---

## 6. 训练结果回传与查看

### 6.1 训练完成后下载结果

训练结果通常包含：
- 训练好的模型（`point_cloud/` 目录下的 `.ply` 文件）
- 渲染图片（`test/` 目录）
- 指标文件（`results.json`）

```bash
# 从远程下载结果到本地
mkdir -p ~/mvgs/results
rsync -avz --progress -e 'ssh -p 12345' \
    root@connect.bjb1.autodl.com:/root/autodl-tmp/output/garden/ \
    ~/mvgs/results/garden/
```

> 用 Open Interpreter 的话，直接说：
> ```
> 请帮我把远程 /root/autodl-tmp/output/garden/ 下载到本地 ~/mvgs/results/garden/
> ```

### 6.2 通过 Open Interpreter 查看训练指标

```
请查看本地 ~/mvgs/results/garden/ 下的 results.json 文件，
告诉我 PSNR、SSIM 和 LPIPS 指标。
```

### 6.3 AutoDL 的文件传输替代方案

AutoDL 还提供了**网页版文件管理器**，可以在控制台直接上传/下载文件：

1. 登录 AutoDL 控制台
2. 点击实例 → 「JupyterLab」
3. 在 JupyterLab 中直接上传/下载文件

这种方式更直观，但对大文件速度较慢。

---

## 7. 安全提醒

### ⚠️ 密码安全

- **不要将密码硬编码在提交到 Git 的文件中**
- Open Interpreter 会在本地执行命令，密码可能出现在 shell 历史中
- 训练完成后建议清除历史：`history -c`

### ⚠️ AutoDL 计费

- AutoDL 按**实例运行时间**计费，不是按 GPU 使用时间
- **训练完成后立即关机**，否则持续产生费用
- 可以使用 Open Interpreter 关机：

  ```
  训练完成后，请在远程服务器执行关机命令：shutdown now
  ```

  或在 AutoDL 控制台手动关机。

### ⚠️ Open Interpreter 权限

Open Interpreter 默认会请求你确认每个命令。**保持这个确认机制开启**，
不要使用 `--auto_run` 模式，以免误操作。

---

## 8. 常见问题

### Q1: Open Interpreter 报错 `sshpass: command not found`

```bash
sudo apt install -y sshpass
```

### Q2: SSH 连接超时 `Connection timed out`

- 确认 AutoDL 实例已**开机运行**（不是关机状态）
- 确认端口号正确（AutoDL 每次开机端口可能变化）
- 在 AutoDL 控制台刷新获取最新的 SSH 信息

### Q3: 上传速度很慢

- 如果从 Windows 路径（`/mnt/c/`）上传很慢，先复制到 WSL 本地：
  ```bash
  cp -r /mnt/c/path/to/data ~/mvgs/data
  ```
- 使用 `rsync` 代替 `scp`，支持增量传输和断点续传
- 大数据集考虑使用 AutoDL 的「数据盘上传」功能或挂载网盘

### Q4: 远程编译 CUDA 扩展失败

参考 [AUTODL_SETUP_GUIDE.md](AUTODL_SETUP_GUIDE.md) 第 5 节中的详细排查步骤。

最常见原因：
- CUDA 版本不匹配 GPU → 参考镜像选择对照表
- 子模块未下载 → 手动上传 `submodules/` 目录的完整内容

### Q5: 训练中断了，如何恢复？

本项目支持从 checkpoint 恢复训练：

```bash
python train.py -s {data_path} --eval --white_background \
    -m {save_path} --mv 4 \
    --start_checkpoint {save_path}/chkpnt30000.pth
```

### Q6: Open Interpreter 不认识 `rsync` 命令的结果

Open Interpreter 有时会对长输出截断。可以将命令输出重定向到文件：

```
请执行上传命令，并把输出保存到 /tmp/upload.log，完成后告诉我最后 5 行。
```

### Q7: 如何让 Open Interpreter 记住 SSH 信息？

在对话开始时设定系统消息：

```
从现在开始，所有远程操作都使用以下 SSH 信息：
主机: connect.bjb1.autodl.com，端口: 12345，用户: root，密码: xxxxxxxx
每次执行远程命令时都用 sshpass 和这些信息。不需要每次都问我。
```

---

## 附录：Open Interpreter 常用提示词速查

| 任务 | 提示词 |
|---|---|
| 测试连接 | `请用 SSH 连接远程服务器并运行 nvidia-smi` |
| 上传文件 | `请用 rsync 上传 ~/mvgs/code/ 到远程 /root/autodl-tmp/MVGS/` |
| 安装依赖 | `请在远程服务器上安装训练依赖` |
| 启动训练 | `请在远程后台启动训练，用 nohup 和 &` |
| 查看日志 | `请查看远程训练日志最后 30 行` |
| 查看 GPU | `请查看远程 GPU 使用情况` |
| 下载结果 | `请把远程训练结果下载到本地` |
| 关机省钱 | `训练完成了，请远程关机` |
