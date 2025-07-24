# GeometryCrafter (Jittor Version)

This repository is a **Jittor-based reimplementation** of [TencentARC/GeometryCrafter](https://github.com/TencentARC/GeometryCrafter), originally written in PyTorch. GeometryCrafter is a framework for controllable geometry-oriented video generation, supporting point-cloud and depth-aware synthesis from text and video prompts.

In this version, we migrate the main inference pipeline to [Jittor](https://github.com/Jittor/jittor) for high-performance and flexible execution on research platforms where Jittor is preferred or required.

To ensure compatibility with the original [MoGe](https://github.com/microsoft/MoGe) prior module (which is tightly coupled to PyTorch), we run it in a separate Conda environment and communicate through a subprocess interface.

> ⚙️ This dual-environment design ensures Jittor and PyTorch can coexist without dependency conflicts, especially across CUDA versions.

---
## 📁 Directory Overview

```
GeometryCrafter/
├─ geometrycrafter/            # Jittor pipeline
├─ third_party/moge/           # MoGe module (uses PyTorch)
├─ requirements_jt.txt         # Requirements for Jittor environment
├─ requirements_torch.txt      # Requirements for PyTorch/MoGe environment
├─ moge_worker.py              # Worker script for MoGe subprocess
├─ run_jt.py                   # Main entry (Jittor + subprocess call)
```

---

## 🧪 Setup Two Environments

### Environment 1: Jittor (named `geo_jt`)

```bash
conda create -n geo_jt python=3.9 -y
conda activate geo_jt

# Prevent loading ~/.local packages
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export PYTHONNOUSERSITE=1' > $CONDA_PREFIX/etc/conda/activate.d/no_user_site.sh

# Install requirements
python -m pip install --no-cache-dir -r requirements_jt.txt
```

### Environment 2: PyTorch for MoGe (named `geo_torch`)

```bash
conda create -n geo_torch python=3.9 -y
conda activate geo_torch

echo 'export PYTHONNOUSERSITE=1' > $CONDA_PREFIX/etc/conda/activate.d/no_user_site.sh
python -m pip install --no-cache-dir -r requirements_torch.txt
```

---

## 🚀 Run Main Pipeline (Jittor side)

```bash
conda activate geo_jt

python run_jt.py \
  --video_path examples/video.mp4 \
  --save_folder workspace/output \
  --height 384 --width 640 \
  --low_memory_usage True \
  --decode_chunk_size 6
```

The pipeline automatically launches the subprocess using:

```bash
conda run -n geo_torch python moge_worker.py /tmp/input.npy
```

## 📝 Protocol: MoGe Worker Communication

- Input: `.npy` file (shape `(1, 3, H, W)`)
- Output: JSON line via stdout like:
  ```json
  {"p": "/tmp/point.npy", "m": "/tmp/mask.npy"}
  ```

---

This dual-environment approach is robust for managing CUDA/toolkit incompatibilities and enables clean modular execution.

Happy GeometryCrafting 🎨
