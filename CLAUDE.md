# ProPainter-Resolve-Node — Claude context

## Project overview
Phase 1: validate the full ProPainter inference stack (WAFT flow → RecurrentFlowCompleteNet → InpaintGenerator) on Python 3.14 / PyTorch 2.12 / CUDA 13.1 / RTX 3080 (8 GB VRAM) before building the DaVinci Resolve OpenFX node in Phase 2.

## Hardware
- **Current**: RTX 3080 8 GB — VRAM-constrained; `subvideo_length`, FP16 (ProPainter), FP16 WAFT, and TurboQuant KV compression are the primary knobs.
- **Pending**: NVIDIA P40 24 GB — will remove most VRAM constraints; see upgrade notes below.

## Repository layout
```
tkinter_app.py              Phase 1 benchmarking harness (3-tab Tkinter UI)
inference_propainter.py     Core inference helpers (resize_frames, read_mask, get_ref_index,
                            extrapolation) — do NOT import fix_raft/RAFT_bi from here;
                            all flow goes through _models["flow"] in tkinter_app.py
model/
  propainter.py             InpaintGenerator
  recurrent_flow_completion.py  RecurrentFlowCompleteNet
  modules/
    flow_comp_waft.py       WAFT_bi — optical flow via WAFT (scoped sys.modules swap);
                            targets waft-a1/dav2 (ViTWarpV8); WAFT-twins path present
                            but no downstream checkpoint exists
    sparse_transformer.py   SparseWindowAttention (TurboQuant hook lives here)
    turboquant_kv.py        TurboQuant KV cache (Lloyd-Max + QJL residual)
WAFT/                       Git submodule — WAFT optical flow model (waftv2 branch)
weights/                    Downloaded checkpoints (gitignored)
results/                    Output videos — {stem}_{run_label}_inpainted.mp4
                            PNG sequences — {stem}_{run_label}_frames/ (when save_frames=True)
core/utils.py               to_tensors(), Stack, ToTorchFormatTensor (no torchvision)
```

## Key design decisions
- **No torchvision**: eliminated; `to_tensors()` is a plain lambda over `Stack` + `ToTorchFormatTensor`.
- **WAFT/ProPainter namespace isolation**: `flow_comp_waft.py` temporarily evicts `model.*` from `sys.modules` before importing WAFT's `model.waft_a1`, then restores. Avoids package collision without permanent `sys.path` mutation.
- **Single flow model path**: ALL flow computation in `tkinter_app.py` goes through `_models["flow"]` (a `WAFT_bi` instance). `inference_propainter.py`'s internal `fix_raft` is never used; `RAFT_bi` is never imported directly. The Flow Backbone combobox determines what `_models["flow"]` is — this applies to all three modes.
- **TurboQuant**: optional KV cache compression (Lloyd-Max quantization + QJL residual correction) controlled by `use_turboquant` flag on `SparseWindowAttention`. Off by default — zero behavior change.
- **FP16 WAFT**: optional half-precision cast of the WAFT flow model (`fp16=True` on `WAFT_bi`). Independent of ProPainter FP16. DA2's attention is per-frame (not a persistent KV cache), so TurboQuant on DA2 would be overhead with no benefit — FP16 is the right VRAM knob for WAFT.
- **WAFT CPU offload**: after flow is computed each chunk, `flow_model.cpu()` + `empty_cache()` frees VRAM for the heavier flow completion + inpainting pass; `flow_model.to(device)` restores it at the next chunk start.
- **4K tiling**: `WAFT_bi._tile_flow()` splits frames > 1080p into a 2×2 overlapping grid (64px overlap, linear blend), runs WAFT on each tile, stitches back. Threshold: H > 1080 or W > 1920.
- **inference_mode**: all inference contexts use `torch.inference_mode()` instead of `torch.no_grad()` — disables autograd more aggressively, reduces per-tensor overhead.
- **cudnn benchmark**: `torch.backends.cudnn.benchmark = True` set once at run start for consistent input sizes.
- **Error recovery**: entire `run_inpainting()` body wrapped in `try/except`; `done_fn(None, None, {})` always called so the Run button re-enables even on crash. Full traceback logged to the UI log widget.
- **Flow model cache eviction**: `_models` tracks `"flow_backbone"` and `"flow_fp16"`; any change evicts the old model and calls `empty_cache()` before reloading.
- **Three inference modes**: Video Inpainting (mask-based), Video Outpainting (`extrapolation()` generates expanded canvas + masks automatically, no mask path needed), Resolution Expansion (resize then full-frame mask so ProPainter regenerates detail via temporal propagation).
- **Dynamic UI**: `_build_run_left` uses three `ttk.Frame` objects at the same grid row; `_on_mode_changed` calls `grid_remove()`/`grid()` to swap the visible section without destroying widgets or losing state.
- **Direct file paths throughout**: `tkinter_app.py` uses `askopenfilename`; no web server, no temp files, no Gradio.
- **Output naming**: `{video_stem}_{run_label}_inpainted.mp4` — runs never overwrite each other, enabling A/B comparison. PNG sequences go to `{stem}_{label}_frames/` when save_frames is enabled.

## Claude working notes — things to check before editing
- **Always run `Select-String -Path tkinter_app.py -Pattern "fix_raft|RAFT_bi"` (or Grep) before touching flow code** — there must be zero matches. All flow goes through `_models["flow"]`.
- **WAFT-dav2 is the confirmed primary target** — `waft-downstream.pth` is a waft-a1/dav2 checkpoint with DA2 baked in. Do not add InferenceWrapper or external DA2 loading logic.
- **WAFT-twins is dead code** — no downstream checkpoint exists. The path in `flow_comp_waft.py` is retained for when one is released.
- **`_on_chunks_changed` disables `_sv_spin`** when chunks > 0. When editing spinbox state logic, always handle the `TclError`/`ValueError` guard (spinbox trace fires during partial typing).
- **Three mode-specific frames share the same grid row** (`MODE_FRAME_ROW`). Use `grid_remove()`/`grid()` not `pack_forget()`/`pack()` — they were laid out with grid.
- **`run_inpainting` param order**: `video_path, mask_path, mode, run_label, neighbor_length, ref_stride, subvideo_length, fp16, use_tq, tq_bits, mask_dilation, flow_backbone, fp16_waft, num_chunks, scale_h, scale_w, resize_ratio, res_width, res_height, save_fps, save_frames, raft_iters, log_fn, progress_fn, vram_sample_fn, done_fn`
- **pip install timm requires --no-deps** — without it pip pulls torch from PyPI and destroys the custom wheel.

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