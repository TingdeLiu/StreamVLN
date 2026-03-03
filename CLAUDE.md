# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

StreamVLN is a streaming Vision-and-Language Navigation (VLN) framework built on top of **LLaVA-Video** (Qwen2-based). It uses a SlowFast context modeling approach: a **fast-streaming** sliding-window KV cache for recent frames, and a **slow-updating** memory via voxel-based spatial token pruning for long-range history. The model treats navigation actions (↑, ←, →, STOP) as text tokens in an autoregressive interleaved dialogue.

## Environment Setup

```bash
conda create -n streamvln python=3.9
conda install habitat-sim==0.2.4 withbullet headless -c conda-forge -c aihabitat
git clone --branch v0.2.4 https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab && pip install -e habitat-lab && pip install -e habitat-baselines
pip install -r requirements.txt
```

Dependencies: Python 3.9, PyTorch 2.1.2, CUDA 12.4, habitat-sim 0.2.4, transformers 4.45.1, deepspeed 0.14.4, flash-attention-2.

## Key Commands

**Stage-1 Training** (multi-node SLURM):
```bash
sbatch scripts/streamvln_train_slurm.sh
```

**DAgger Data Collection**:
```bash
sh scripts/streamvln_dagger_collect.sh
```

**Stage-2 Co-training** (multi-node SLURM):
```bash
sbatch scripts/streamvln_stage_two_train_slurm.sh
```

**Evaluation** (multi-GPU):
```bash
sh scripts/streamvln_eval_multi_gpu.sh
# Equivalent to:
torchrun --nproc_per_node=8 --master_port=$MASTER_PORT streamvln/streamvln_eval.py --model_path $CHECKPOINT
```

**Trajectory Data Generation**:
```bash
sh scripts/streamvln_trajectory_generation.sh
```

All scripts set `MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet` to suppress simulator noise. DeepSpeed configs are in `scripts/zero2.json` (stage-1) and `scripts/zero3.json` (stage-2).

## Architecture

### Module Responsibilities

| Path | Role |
|------|------|
| `streamvln/model/stream_video_vln.py` | `StreamVLNForCausalLM` — core model extending `Qwen2ForCausalLM` + `LlavaMetaForCausalLM`; contains `get_2dPool()` for spatial pooling of memory tokens |
| `streamvln/streamvln_eval.py` | `VLNEvaluator` for Habitat simulation; manages KV-cache state, calls `parse_actions()` to map LLM text → discrete actions |
| `streamvln/streamvln_agent.py` | `VLNEvaluator` for real-world robot deployment (reuses same class name, different sensor interface) |
| `streamvln/streamvln_train.py` | Training entry point |
| `streamvln/streamvln_dagger.py` | DAgger interactive data collection loop |
| `streamvln/streamvln_trajectory_generation.py` | Offline trajectory collection from expert policy |
| `streamvln/dataset/vln_action_dataset.py` | `VLNActionDataset` — builds interleaved `<image>`/`<memory>`/action token sequences |
| `streamvln/dataset/mmc4_dataset.py` | MMC4 co-training dataset |
| `streamvln/args.py` | `ModelArguments`, `DataArguments`, `TrainingArguments` dataclasses |
| `streamvln/utils/` | Shared utilities and distributed helpers |
| `llava/` | LLaVA-NeXT codebase (vision encoder, projector, LlavaQwen model, conversation templates) |
| `config/` | Habitat YAML configs (`vln_r2r.yaml`, `vln_dagger.yaml`, `co-training_data.yaml`) |
| `realworld/` | Physical robot deployment: `go2_vln_client.py`, `pid_controller.py` for Unitree Go2 |

### Import Convention

Scripts in `streamvln/` add the project root to `sys.path` at runtime:
```python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
```
They then import directly as `from model.stream_video_vln import StreamVLNForCausalLM` (not `from streamvln.model...`).

### Key Design Parameters

- `num_history`: number of pooled memory frames inserted as `<memory>` tokens (slow context)
- `num_future_steps`: number of action steps predicted per forward pass
- `num_frames`: total frames sampled per dialogue turn (default 32)
- Memory tokens use 2D average/max/bilinear pooling (`mm_spatial_pool_mode`) with stride 2 by default
- Context window: `model_max_length=32768` for training; attention via Flash Attention 2

### Action Space

Actions are Unicode arrows in the tokenizer vocabulary:
- `↑` → forward 25 cm
- `←` → left turn 15°
- `→` → right turn 15°
- `STOP` → terminate episode

### Data Layout

```
data/
├── datasets/         # VLN-CE episode JSONs (r2r/, rxr/, envdrop/, scalevln/)
├── scene_datasets/   # mp3d/ (R2R/RxR/EnvDrop) and hm3d/ (ScaleVLN)
├── trajectory_data/  # Pre-collected observation-action pairs (R2R/, RxR/, EnvDrop/, ScaleVLN/)
├── dagger_data/      # DAgger-collected correction data
└── co-training_data/ # ScanNet/, LLaVA-Video-178K/, MMC4-core/
```

### Base Model

Training starts from `lmms-lab/LLaVA-Video-7B-Qwen2` (LLM: `Qwen/Qwen2-7B-Instruct`, vision encoder: `google/siglip-so400m-patch14-384`, projector: `mlp2x_gelu`).

Published checkpoints:
- Benchmark: `mengwei0427/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3`
- Real-world: `mengwei0427/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_real_world`

## Training Data Versions

The repository uses **R2R_VLNCE_v1-3** (not v1) as of the Sep 2025 checkpoint update. Make sure `data/datasets/r2r/` contains v1-3 episodes, and `data/trajectory_data/R2R_V1-3/` annotations if running DAgger.
