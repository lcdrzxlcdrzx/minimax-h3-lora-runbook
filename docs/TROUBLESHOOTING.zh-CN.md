# 故障排查

## INT8 ConvRot 反向 CUBLAS 失败

**现象：**前向成功，反向在 `torch._int_mm` 附近报 cuBLAS unsupported-status 错误。

**处理：**查看[上游 issue #1059](https://github.com/kohya-ss/musubi-tuner/issues/1059)，并确认当前 musubi-tuner revision。发布的 pruned checkpoint 可能包含不满足 INT8 GEMM 对齐要求的维度。在确认当前 revision 实际选择的 operation 前，不要假设 CLI 的 BF16 反向参数一定生效。

## 验证集数量大于 JSONL 划分

**现象：**日志显示 JSONL 加载了 N 条，但 bucket 数量或 `num train items` 更大。

**处理：**检查配置的 cache 目录里是否有不属于该 split 的文件。训练和验证的 latent/text pair 必须放到独立目录。详见[上游 issue #1060](https://github.com/kohya-ss/musubi-tuner/issues/1060)。

## Block swap 反向出现 BF16/FP32 dtype 不匹配

**现象：**流式冻结层报告 BF16/FP32 matmul 类型不匹配。

**处理：**先更新 musubi-tuner。若当前 revision 仍可复现，记录完整 traceback、commit/version、最小 tensor shape/dtype 和 1-step 复现，再考虑本地 patch。不要把针对旧源码结构的 patch 直接套到新版本。

## OOM

降低冒烟测试分辨率或帧数，逐步提高 `blocks_to_swap`，保持 batch size 1 并保留 gradient checkpointing。一次只改一个变量，然后重新跑 1-step。

## 训练太慢

跳过首几步加载阶段后，记录稳定的每步秒数。Block swap 可以让模型“装得下”，但训练可能被主机到显卡的数据传输主导。更大显存可能降低总成本，即使小时价格更高；但必须用真实 1-step/10-step benchmark 比较。

## 中断后恢复

不要假设 output 目录中就一定存在完整 state。检查模型、优化器、scheduler、sampler 和 random-state 文件，再做一次真实 resume 测试，之后才开始长租训练。
