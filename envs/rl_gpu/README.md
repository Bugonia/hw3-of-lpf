# Offline RL GPU Environment

This directory is a project-local launcher for the shared offline `hw3`
environment:

```text
/inspire/hdd/project/generative-large-model/public/envs/hw3
```

Use it on any machine that shares the project filesystem:

```bash
cd /inspire/hdd/project/generative-large-model/public/ywy/hw3-of-lpf
source envs/rl_gpu/activate.sh
bash scripts/preflight_rl_gpu.sh
```

The activation script sets offline flags and cache paths so the training scripts
do not try to access the network.

Expected package metadata in the shared env:

```text
torch 2.11.0
transformers 5.13.0.dev0
peft 0.19.1
vllm 0.23.0
accelerate 1.14.0
bitsandbytes 0.49.2
```

`trl` is not required. The DPO-style trainer in this repo is implemented with
PyTorch, Transformers, and PEFT directly.
