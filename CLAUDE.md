# ProPainter-Resolve-Node — Claude context

## Project overview
Phase 1: validate the full ProPainter inference stack (WAFT flow → RecurrentFlowCompleteNet → InpaintGenerator) on Python 3.14 / PyTorch 2.12 / CUDA 13.1 / RTX 3080 (8 GB VRAM) before building the DaVinci Resolve OpenFX node in Phase 2.

## Hardware
- **Current**: RTX 3080 8 GB — VRAM-constrained; `subvideo_length`, FP16, and TurboQuant KV compression are the primary knobs.
- **Pending**: NVIDIA P40 24 GB — will remove most VRAM constraints; see upgrade notes below.

## Repository layout
```
tkinter_app.py              Phase 1 benchmarking harness (3-tab Tkinter UI)
inference_propainter.py     Core inference helpers (resize_frames, read_mask, get_ref_index)
model/
  propainter.py             InpaintGenerator
  recurrent_flow_completion.py  RecurrentFlowCompleteNet
  modules/
    flow_comp_waft.py       WAFT_bi — optical flow via WAFT (scoped sys.modules swap)
    sparse_transformer.py   SparseWindowAttention (TurboQuant hook lives here)
    turboquant_kv.py        TurboQuant KV cache (Lloyd-Max + QJL residual)
WAFT/                       Git submodule — WAFT optical flow model
weights/                    Downloaded checkpoints (gitignored)
results/                    Output videos — {stem}_{run_label}_inpainted.mp4
core/utils.py               to_tensors(), Stack, ToTorchFormatTensor (no torchvision)
```

## Key design decisions
- **No torchvision**: eliminated; `to_tensors()` is a plain lambda over `Stack` + `ToTorchFormatTensor`.
- **WAFT/ProPainter namespace isolation**: `flow_comp_waft.py` temporarily evicts `model.*` from `sys.modules` before importing WAFT's `model.waft_a1`, then restores. Avoids package collision without permanent `sys.path` mutation.
- **TurboQuant**: optional KV cache compression (Lloyd-Max quantization + QJL residual correction) controlled by `use_turboquant` flag on `SparseWindowAttention`. Off by default — zero behavior change.
- **Direct file paths throughout**: `tkinter_app.py` uses `askopenfilename`; no web server, no temp files, no Gradio.
- **Output naming**: `{video_stem}_{run_label}_inpainted.mp4` — runs never overwrite each other, enabling A/B comparison.

## Phase 2 plan
DaVinci Resolve OpenFX node (C++ plugin) that calls into the Python inference stack. Phase 1 must be validated first.

---

## P40 upgrade notes

### DiffuEraser
- **What**: Diffusion-based video inpainting model; uses ProPainter as a prior/initialization and applies a video diffusion model on top.
- **Why relevant**: Explicitly outperforms ProPainter on temporal consistency benchmarks (DAVIS, YouTube-VOS). The diffusion pass smooths temporal flickering that ProPainter's propagation+transformer approach leaves behind.
- **Weights**: ~30 GB (SD-based video diffusion backbone). Not viable on RTX 3080 8 GB.
- **Revisit when**: P40 (24 GB) arrives. DiffuEraser would replace or wrap the current `InpaintGenerator` inference path.
- **Reference**: "DiffuEraser: Diffusion Model for Video Inpainting" — integrates ProPainter optical flow + propagation as a structured prior before the diffusion U-Net.
