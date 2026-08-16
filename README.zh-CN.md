# MiniMax H3 LoRA 空白服务器部署 Runbook

[English](README.md) | [快速开始](#快速开始) | [故障排查](docs/TROUBLESHOOTING.zh-CN.md)

这是一个用于在全新 NVIDIA GPU 服务器上部署 **MiniMax H3 Ref2VA-Omni 视频 LoRA 训练**的通用 Runbook，训练框架为 [musubi-tuner](https://github.com/kohya-ss/musubi-tuner)。

它来自一次完整的受限硬件验证：先在 Windows + 20GB GPU 上完成 1-step 冒烟测试，再在 Linux + 48GB GPU 上完成 768x1344 缓存与正式训练准备。目标不是把某套参数说成通用答案，而是把昂贵、未知的长训练失败转换为廉价、可检查的前置步骤。

## 范围与安全边界

本仓库不包含数据集、提示词、模型权重、checkpoint、服务器地址、凭据或任何平台账号信息。

包含内容：

- 隔离 Python/CUDA 环境验证；
- 模型文件完整性校验；
- 视频 JSONL 数据规则；
- latent/text cache 验收；
- 1-step 与 10-step 冒烟测试；
- 训练/验证 cache 隔离；
- checkpoint/resume 纪律；
- TensorBoard 监控与 loss 解读；
- 显存 block swap 调参。

本仓库不直接附带 musubi-tuner 源码补丁。上游版本会变化，应用 workaround 前应先查看相关上游 issue 状态。

## 快速开始

1. 阅读 musubi-tuner 的官方 MiniMax H3 文档，安装其支持的 Python/PyTorch/CUDA 组合。
2. 运行环境探针：

   ```bash
   python scripts/verify_environment.py
   ```

   只有 CUDA 可用且 BF16 tensor 分配通过后才继续。

3. 缓存前校验每个模型：

   ```bash
   python scripts/verify_safetensors.py /path/to/model.safetensors --sha256 官方_SHA256 --bytes 官方字节数
   ```

4. 创建每行一个目标的 JSONL，示例见 [examples/video-jsonl.example.jsonl](examples/video-jsonl.example.jsonl)。
5. 训练集使用 `examples/dataset-train.example.toml` 建 cache；验证集必须使用独立目录与 `examples/dataset-val.example.toml`。
6. 先缓存 latent，再缓存文本编码输出；确认每个目标都有一份 latent cache 和一份 text cache。
7. 先跑 `max_train_steps=1`，再跑 `max_train_steps=10`。验证前向、反向、优化器更新、LoRA 保存和完整 Accelerate state 保存。
8. 最后才用 `examples/train.example.sh` 作为参数模板启动长训练。

## 不可跳过的检查

### 1. 不要只相信下载器的成功提示

下载器可能显示成功，但旧控制文件或中断传输会留下不可用的 safetensors。每个模型都做三项校验：

- 精确预期字节数；
- 官方 SHA-256；
- safetensors 实际可读性。

大文件可以使用可恢复多连接下载，但必须验收最终文件：

```text
--continue=true --max-connection-per-server=16 --split=16 --min-split-size=20M --file-allocation=none
```

### 2. 清空继承的 PYTHONPATH

继承的 `PYTHONPATH` 可能从其他 Python 应用加载不兼容依赖：

```bash
env -u PYTHONPATH python your_command.py
```

Windows `.bat` 启动器应加：

```bat
set "PYTHONPATH="
```

### 3. 绝不跳过 1-step

合格的 1-step 必须完整走完：

- transformer 前向；
- 反向；
- optimizer 更新；
- LoRA 权重保存；
- 完整 resume state 保存。

只看到进度条不代表训练链路成功。

### 4. 训练和验证必须独立 cache 目录

不能只靠不同 JSONL 文件划分训练/验证。如果 cache 构建或 bucket 阶段枚举整个目录，验证样本可能进入训练。

```text
cache/train/   # 只包含训练 latent/text cache pair
cache/val/     # 只包含验证 latent/text cache pair
```

启动训练时检查日志：`num train items` 和 batch 数必须等于训练集预期数，不能等于全部 cache 数。

详见上游 issue：[ #1060 ](https://github.com/kohya-ss/musubi-tuner/issues/1060)。

### 5. 高频保存完整状态

可中断或预付费租卡应保存完整 state，而不只是 LoRA：

```text
--save_every_n_steps 20
--save_state
--autoresume
```

正式长训前做一次真实 resume 验证。可用的状态应包含模型、优化器、scheduler、dataloader sampler 与随机状态。

## 显存起测档位

下表是起测原则，不是保证。必须按实际 GPU、模型版本、分辨率和帧数测量。

| 档位 | 用途 | 起测策略 |
|---|---|---|
| 20GB | 只验证全流程 | batch 1、rank 16、gradient checkpointing、高 block swap，只跑 1-step |
| 48GB | 768x1344 / 73 帧准备 | batch 1、gradient checkpointing，先从 `blocks_to_swap=24` 测起，不够再提高 |
| 80GB+ | 更快迭代 / 更大数据 | 从低/零 block swap 开始，实测显存和吞吐 |

某次 48GB、768x1344 / 73 帧的真实配置中，`blocks_to_swap=40` 能稳定运行，但每个优化步约 123 秒。这只是该配置的实测数据，不是显卡通用 benchmark。

## 监控与决策

启动 TensorBoard：

```bash
tensorboard --logdir logs/tb --host 0.0.0.0 --port 6006
```

主要指标：

- `loss/current`：噪声很大的单 batch 值；
- `loss/average`：当前进程内累计均值，恢复训练开新 run 后会重新累计；
- `val/loss`、`val/loss/video`、`val/loss/audio`：判断过拟合最重要；
- `grad/norm`：检测非法或爆炸梯度。

Flow-matching loss 没有跨数据集通用的绝对及格线，重点看趋势：

- train loss 降且 validation loss 降/稳定：继续；
- train loss 降而 validation loss 连续上升：停止或选择更早 checkpoint；
- NaN/Inf、持续梯度异常、验证失败：先排查再继续花费。

小视频数据集应在多个 checkpoint 用固定提示词和固定 seed 采样对比，不要默认最终 checkpoint 最好。

## 已知上游问题与临时处理

- [#1059](https://github.com/kohya-ss/musubi-tuner/issues/1059)：Ref2VA pruned INT8 ConvRot 反向可能选择不兼容的 INT8 GEMM 路径，即使设置了 BF16 反向选项。应用任何本地 workaround 前先查 issue 状态。
- [#1060](https://github.com/kohya-ss/musubi-tuner/issues/1060)：训练/验证共用 cache 目录可能导致验证样本泄露进训练 bucket。上游修复未确认前请使用独立 cache 目录。

## 仓库结构

```text
examples/      通用 TOML、JSONL 和训练参数模板
scripts/       只读环境与 safetensors 校验工具
docs/          中英文操作细节和故障排查
README.md      英文主文档
README.zh-CN.md 中文主文档
```

## 贡献

贡献必须可复现且已脱敏。不要提交数据集、提示词、私有模型输出、凭据、服务器 URL、token 或云平台账号信息。详见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 协议

MIT，见 [LICENSE](LICENSE)。
