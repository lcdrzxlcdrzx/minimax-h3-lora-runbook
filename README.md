# MiniMax H3 LoRA Runbook

[中文文档](README.zh-CN.md) | [Quick start](#quick-start) | [Troubleshooting](docs/TROUBLESHOOTING.md)

A vendor-neutral runbook for bringing up **MiniMax H3 Ref2VA-Omni video LoRA training** with [musubi-tuner](https://github.com/kohya-ss/musubi-tuner) on a fresh NVIDIA GPU server.

It is based on an end-to-end validation on constrained hardware: Windows + 20GB GPU for a one-step smoke test, then Linux + 48GB GPU for cached 768x1344 training. The goal is not to prescribe one universal configuration. The goal is to turn expensive unknown failures into cheap, measurable checks.

## Scope and safety

This repository contains no dataset, prompts, model weights, checkpoints, server addresses, credentials, or provider-specific account details.

It covers:

- isolated Python/CUDA validation;
- model-file integrity checks;
- video JSONL preparation rules;
- latent/text cache verification;
- one-step and ten-step smoke tests;
- train/validation cache isolation;
- checkpoint/resume discipline;
- TensorBoard monitoring and loss interpretation;
- VRAM block-swap tuning.

It does **not** ship musubi-tuner source patches. Upstream code evolves; consult the linked upstream issues before applying a workaround.

## Quick start

1. Read the official MiniMax H3 documentation in musubi-tuner and install a supported Python/PyTorch/CUDA combination.
2. Run the environment probe:

   ```bash
   python scripts/verify_environment.py
   ```

   Continue only when CUDA is available and BF16 tensor allocation succeeds.

3. Verify all model files before caching:

   ```bash
   python scripts/verify_safetensors.py /path/to/model.safetensors --sha256 EXPECTED_SHA256 --bytes EXPECTED_BYTES
   ```

4. Create an input JSONL with one target per line. See [examples/video-jsonl.example.jsonl](examples/video-jsonl.example.jsonl).
5. Use `examples/dataset-train.example.toml` to create caches for the training split. Use a separate directory and `examples/dataset-val.example.toml` for validation.
6. Cache latents, then text encoder outputs. Confirm every target has one latent cache and one text cache.
7. Run `max_train_steps=1`, then `max_train_steps=10`. Verify forward, backward, optimizer update, LoRA save, and full Accelerate state save.
8. Only then begin a long run using `examples/train.example.sh` as a parameter template.

## Non-negotiable checks

### 1. Validate model files beyond downloader success

A downloader can report success while a stale control file or interrupted transfer leaves an unusable safetensors file. Check all three:

- exact expected byte count;
- official SHA-256;
- actual safetensors readability.

Use multi-connection resumable downloads only when you also verify the final artifact. A conservative aria2 pattern is:

```text
--continue=true --max-connection-per-server=16 --split=16 --min-split-size=20M --file-allocation=none
```

### 2. Clear inherited Python paths

An inherited `PYTHONPATH` can import packages from another application instead of the intended virtual environment.

```bash
env -u PYTHONPATH python your_command.py
```

Windows batch launchers should include:

```bat
set "PYTHONPATH="
```

### 3. Never skip the one-step test

A valid one-step smoke test must complete all of the following:

- transformer forward pass;
- backward pass;
- optimizer update;
- LoRA weight save;
- complete resume state save.

A visible progress bar alone is not proof that training works.

### 4. Isolate training and validation cache directories

Use distinct cache directories for train and validation samples. Do not rely on separate JSONL files alone when the cache builder/trainer can enumerate a whole cache directory.

```text
cache/train/   # only training latent/text cache pairs
cache/val/     # only validation latent/text cache pairs
```

At startup, check the log: `num train items` and the number of batches must match the intended training split, not the total cached item count.

See upstream report: [#1060](https://github.com/kohya-ss/musubi-tuner/issues/1060).

### 5. Save complete state frequently

For interruptible or prepaid rentals, save complete state rather than only LoRA weights:

```text
--save_every_n_steps 20
--save_state
--autoresume
```

Before a long run, verify one real resume. A usable state contains model, optimizer, scheduler, dataloader sampler, and random-state files.

## VRAM profiles

These are starting points, not guarantees. Always measure on the actual GPU, model build, resolution, and frame count.

| Profile | Intended use | Starting strategy |
|---|---|---|
| 20GB | pipeline validation only | batch 1, rank 16, gradient checkpointing, high block swap; run 1 step |
| 48GB | 768x1344 / 73-frame preparation | batch 1, gradient checkpointing, begin around `blocks_to_swap=24`, raise only if OOM |
| 80GB+ | faster iteration / larger data | begin with low or zero block swap, then measure VRAM and throughput |

On one measured 48GB configuration at 768x1344 / 73 frames, `blocks_to_swap=40` ran stably but took about 123 seconds per optimizer step. Treat this as a configuration-specific datapoint, not a hardware benchmark.

## Monitoring and decision rules

Run TensorBoard against the training log directory:

```bash
tensorboard --logdir logs/tb --host 0.0.0.0 --port 6006
```

Interpret the signals as follows:

- `loss/current`: noisy single-batch value;
- `loss/average`: process-local average that can restart after a resume;
- `val/loss`, `val/loss/video`, `val/loss/audio`: primary overfitting signal;
- `grad/norm`: detect invalid or exploding gradients.

Flow-matching loss has no portable absolute pass threshold. Prefer trends:

- train loss down and validation loss down/stable: continue;
- train loss down while validation loss rises across checkpoints: stop or select an earlier checkpoint;
- NaN/Inf, persistent gradient abnormality, or validation failure: diagnose before spending more GPU time.

For small video datasets, generate fixed-prompt, fixed-seed samples at several checkpoints. Do not assume the final checkpoint is best.

## Known upstream issues and workarounds

- [#1059](https://github.com/kohya-ss/musubi-tuner/issues/1059): a Ref2VA pruned INT8 ConvRot backward path can select an unsupported INT8 GEMM route despite a BF16 backward option. Check issue status before applying any local source workaround.
- [#1060](https://github.com/kohya-ss/musubi-tuner/issues/1060): shared train/validation cache directories can leak validation samples into training buckets. Use isolated cache directories until an upstream fix is confirmed.

## Repository layout

```text
examples/      Generic dataset and training templates
scripts/       Read-only environment and safetensors verification helpers
docs/          Detailed bilingual operating notes and troubleshooting
README.md      English guide
README.zh-CN.md Chinese guide
```

## Contributing

Contributions should be reproducible and sanitized. Do not submit datasets, prompts, private model outputs, credentials, server URLs, access tokens, or provider account information. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT. See [LICENSE](LICENSE).
