# GeometryCrafter: Dual Environment Setup (Jittor + PyTorch for MoGe)

This project separates the Jittor inference and the PyTorch-based MoGe prior module into two isolated Conda environments to avoid conflicts in dependencies and CUDA toolkits.

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

## ✅ Test the MoGe Worker

```bash
conda activate geo_torch
python moge_worker.py /path/to/test.npy
```

Expected output:

```
[worker] loading /path/to/test.npy
[worker] shape (1, 3, 64, 64)
{"p": "/tmp/xxx.npy", "m": "/tmp/yyy.npy"}
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

---

## 🧠 Optional: Persistent Subprocess Mode

To avoid repeated startup cost of MoGe model, use a persistent subprocess:

```python
# Inside diff_ppl_jt.py
self._torch_worker = subprocess.Popen(
    ["conda", "run", "-n", "geo_torch", "python", "-u", "moge_worker_long.py"],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1
)
```

Then send JSON through `stdin`, and parse result from `stdout`.

---

## 🧰 Troubleshooting

| Issue | Solution |
|-------|----------|
| Pip installs to `~/.local` | Set `PYTHONNOUSERSITE=1` or use `--no-user` option |
| `jtorch.*` error in Jittor env | Avoid importing PyTorch inside `geo_jt` |
| `JSONDecodeError` or `BrokenPipeError` | Ensure `moge_worker.py` outputs **exactly one line** of valid JSON and flushes it |
| Subprocess exits silently | Use `stderr=subprocess.STDOUT` in `Popen` to capture all logs |

---

## 📝 Protocol: MoGe Worker Communication

- Input: `.npy` file (shape `(1, 3, H, W)`)
- Output: JSON line via stdout like:
  ```json
  {"p": "/tmp/point.npy", "m": "/tmp/mask.npy"}
  ```

---

This dual-environment approach is robust for managing CUDA/toolkit incompatibilities and enables clean modular execution.

Happy GeometryCrafting 🎨
