# Troubleshooting

## INT8 ConvRot backward CUBLAS failure

**Symptom:** forward succeeds, backward fails with a cuBLAS unsupported-status error around `torch._int_mm`.

**Action:** inspect [upstream issue #1059](https://github.com/kohya-ss/musubi-tuner/issues/1059) and check your musubi-tuner revision. The released pruned checkpoint can contain a dimension that is not compatible with the INT8 GEMM alignment requirements. Do not assume that a CLI BF16-backward flag is effective until the selected operation is confirmed in your revision.

## Validation count is larger than the JSONL split

**Symptom:** the log says the JSONL loaded N items, but bucket counts or `num train items` are larger.

**Action:** check whether unrelated cache files are in the configured cache directory. Put train and validation latent/text pairs in separate directories. See [upstream issue #1060](https://github.com/kohya-ss/musubi-tuner/issues/1060).

## BF16/FP32 dtype mismatch during block-swap backward

**Symptom:** a streamed frozen layer reports a BF16/FP32 matmul mismatch.

**Action:** update musubi-tuner first. If the issue is reproducible on the current revision, capture the exact traceback, commit/version, minimal tensor shapes/dtypes, and a 1-step reproduction before considering a local patch. Do not blindly apply a patch written for an older source layout.

## OOM

Reduce resolution or frame count for the smoke test, increase `blocks_to_swap` gradually, keep batch size 1, and retain gradient checkpointing. Change one variable at a time and rerun 1-step.

## Slow training

Record stable seconds/step after the first several steps. Block swap can make a model fit while making training dominated by host-device transfers. More VRAM may improve total cost even when the hourly rate is higher, but compare against a real 1-step/10-step benchmark.

## Resume after interruption

Do not assume an output directory contains a complete state. Check for model, optimizer, scheduler, sampler, and random-state files, then run one real resume test before starting a long rental.
