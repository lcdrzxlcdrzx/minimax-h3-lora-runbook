# Contributing

Thank you for improving this runbook.

## Requirements

- Keep examples provider-neutral and runnable after replacing `${WORKDIR}` and model paths.
- Separate measured facts, local workarounds, and estimates.
- Include GPU, driver, CUDA, PyTorch, musubi-tuner revision, resolution, frames, batch, and relevant swap settings for benchmark claims.
- Add a verification command or acceptance criterion for operational changes.
- Update both English and Chinese documentation when changing user-facing workflow.

## Do not submit

- private datasets, captions, images, videos, LoRA weights, caches, or generated outputs;
- passwords, tokens, SSH keys, server IPs, hostnames, ports, public tunnel URLs, or provider account data;
- logs containing private paths or unrelated user content.

## Pull requests

1. Run `python scripts/verify_environment.py` only when a CUDA environment is available; otherwise run syntax checks.
2. Validate TOML and JSONL examples with the appropriate parser.
3. Run `python -m py_compile scripts/*.py`.
4. Explain what was measured and what remains unverified.
