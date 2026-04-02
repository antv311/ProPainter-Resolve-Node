# ProPainter-Resolve-Node — Claude context

## Project overview
Phase 1: validate the full ProPainter inference stack (WAFT flow → RecurrentFlowCompleteNet → InpaintGenerator) on Python 3.14 / PyTorch 2.12 / CUDA 13.1 / RTX 3080 (8 GB VRAM) before building the DaVinci Resolve OpenFX node in Phase 2.

## Hardware
- **Current**: RTX 3080 8 GB — VRAM-constrained; `subvideo_length`, FP16 (ProPainter), FP16 WAFT, and TurboQuant KV compression are the primary knobs.
- **Pending**: NVIDIA P40 24 GB — will remove most VRAM constraints; see upgrade notes below.

## Repository layout
```
tkinter_app.py              Phase 1 benchmarking harness (3-tab Tkinter UI)
inference_propainter.py     Core inference helpers (resize_frames, read_mask, get_ref_index)
model/
  propainter.py             InpaintGenerator
  recurrent_flow_completion.py  RecurrentFlowCompleteNet
  modules/
    flow_comp_waft.py       WAFT_bi — optical flow via WAFT (scoped sys.modules swap); targets waft-a1/dav2 (ViTWarpV8); WAFT-twins path present but no downstream checkpoint exists
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
- **FP16 WAFT**: optional half-precision cast of the WAFT flow model (`fp16=True` on `WAFT_bi`). Independent of ProPainter FP16. DA2's attention is per-frame (not a persistent KV cache), so TurboQuant on DA2 would be overhead with no benefit — FP16 is the right VRAM knob for WAFT.
- **WAFT CPU offload**: after flow is computed each chunk, `flow_model.cpu()` + `empty_cache()` frees VRAM for the heavier flow completion + inpainting pass; `flow_model.to(device)` restores it at the next chunk start.
- **4K tiling**: `WAFT_bi._tile_flow()` splits frames > 1080p into a 2×2 overlapping grid (64px overlap, linear blend), runs WAFT on each tile, stitches back. Threshold: H > 1080 or W > 1920.
- **inference_mode**: all inference contexts use `torch.inference_mode()` instead of `torch.no_grad()` — disables autograd more aggressively, reduces per-tensor overhead.
- **cudnn benchmark**: `torch.backends.cudnn.benchmark = True` set once at run start for consistent input sizes.
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

**pip install requires --no-deps** — pip install timm without --no-deps will 
pull torch from PyPI and nuke the custom wheel. Always: 
pip install --no-deps timm