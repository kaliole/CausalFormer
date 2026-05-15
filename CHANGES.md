# Changes to CausalFormer

This file documents all modifications made to the original [CausalFormer](https://github.com/lingbai-kong/CausalFormer) repository as part of this thesis project.

Original repo: https://github.com/lingbai-kong/CausalFormer
Fork: https://github.com/kaliole/CausalFormer
Base commit: `19a1b62` ([feat] add dataset files)

---

## Change Log

### 2026-03-13 — Fix MinMaxScaler feature_range type
**File(s):** `data_loader/data_loaders.py`
**What:** Changed `feature_range=[0.5, 1]` (list) to `feature_range=(0.5, 1)` (tuple).
**Why:** scikit-learn >= 1.3 requires `feature_range` to be a tuple. The original code used a list, which raises `InvalidParameterError` with modern scikit-learn.

### 2026-03-13 — Fix torch.load for PyTorch >= 2.6
**File(s):** `interpret.py`, `base/base_trainer.py`
**What:** Added `weights_only=False` to `torch.load()` calls.
**Why:** PyTorch 2.6 changed the default of `weights_only` from `False` to `True`. CausalFormer saves `ConfigParser` objects inside checkpoints, which requires `weights_only=False` to deserialize.

### 2026-05-06 — Add diagonal regularization for self-loop suppression experiment
**File(s):** `model/model.py`, `trainer/trainer.py`, `train.py`
**What:** Added two optional self-loop suppression mechanisms to the attention mask:
- *Soft penalty* (`diag_lam` config key): adds an L2 penalty on the diagonal entries of `MultiVariateCausalAttention.mask` during training, discouraging but not forbidding self-loops. New `diagonal_regularization()` method chain propagates up through `MultiHeadAttention`, `EncoderLayer`, `Encoder`, and `PredictModel`.
- *Hard block* (`diagonal_block: "hard"` config key): zeroes the mask diagonal at init and registers a gradient hook that zeroes diagonal gradients on every backward pass, preventing the model from learning self-loop attention weights.

`Trainer.__init__` gains a `diag_lam=0` parameter; `train.py` reads both `diag_lam` and `diagonal_block` from the trainer config section.
**Why:** Thesis experiment to test whether forcing the model away from autoregressive self-loops improves cross-variable causal discovery quality. Both treatments are optional (default behaviour is unchanged when neither key is set).

### 2026-03-17 — Add Apple MPS device support
**File(s):** `utils/util.py`
**What:** Added MPS (Apple Silicon GPU) as a fallback in `prepare_device()` when CUDA is unavailable but `n_gpu > 0`.
**Why:** The original code only supports CUDA or CPU. On Apple Silicon Macs, MPS enables GPU acceleration without CUDA.

<!--
Format for each entry:

### YYYY-MM-DD — Short description
**File(s):** `path/to/file.py`
**What:** Description of what was changed.
**Why:** Reason for the change.
-->
