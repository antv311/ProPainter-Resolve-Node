"""
ProPainter — Phase 1 Tkinter Benchmarking Harness

Three tabs:
  Run       — configure and execute inference
  Benchmark — VRAM/timing chart + summary table across runs
  Compare   — stub for Phase 1.5 side-by-side video comparison

Output files: results/{stem}_{run_label}_inpainted.mp4
              results/{stem}_{run_label}_comparison.mp4
"""

import os
import sys
import csv
import time
import threading
import tempfile
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext

import cv2
import numpy as np
import imageio
from PIL import Image
from scipy.ndimage import binary_dilation
import torch
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ── project imports ───────────────────────────────────────────────────────────
from inference_propainter import resize_frames, read_mask, get_ref_index, extrapolation
from model.modules.flow_comp_waft import WAFT_bi
from model.recurrent_flow_completion import RecurrentFlowCompleteNet
from model.propainter import InpaintGenerator
from model.modules.sparse_transformer import SparseWindowAttention
from utils.download_util import load_file_from_url
from core.utils import to_tensors
from model.misc import get_device

# ── constants ─────────────────────────────────────────────────────────────────
PRETRAIN_URL = "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/"
WEIGHTS_DIR  = "weights"
RESULTS_DIR  = "results"

_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mts", ".ts"}

# Matplotlib color cycle for per-run lines
_LINE_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]

# ── model registry ─────────────────────────────────────────────────────────────
_model_lock = threading.Lock()
_models: dict = {}
_device: torch.device | None = None


def _get_device() -> torch.device:
    global _device
    if _device is None:
        _device = get_device()
    return _device


def _vram_alloc_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / 1e9


def _vram_str() -> str:
    if not torch.cuda.is_available():
        return "CPU mode (no CUDA)"
    alloc = torch.cuda.memory_allocated() / 1e9
    peak  = torch.cuda.max_memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    return f"Alloc {alloc:.2f} GB  |  Peak {peak:.2f} GB  |  Total {total:.1f} GB"


_BACKBONE_CKPTS = {
    "WAFT-twins": ("waft-downstream.pth", None),   # (filename_in_weights, extra_ckpt_or_None)
    "WAFT-dav2":  ("waft-downstream.pth", "depth-anything-ckpts/depth_anything_v2_vits.pth"),
    "RAFT":       (None, "weights/raft-things.pth"),
    "SEA-RAFT":   (None, "weights/sea-raft-M.pth"),
}

_DAV2_DOWNLOAD_URL = (
    "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/"
    "depth_anything_v2_vits.pth"
)


def load_models(log_fn=None, flow_backbone: str = "WAFT-dav2", fp16_waft: bool = False):
    global _models
    device = _get_device()

    def _log(msg):
        if log_fn:
            log_fn(msg)

    errors = []

    with _model_lock:
        # Evict flow model if backbone or fp16_waft setting has changed since last load
        if (_models.get("flow_backbone") != flow_backbone or
                _models.get("flow_fp16") != fp16_waft):
            _models.pop("flow", None)
            _models.pop("flow_backbone", None)
            _models.pop("flow_fp16", None)
            torch.cuda.empty_cache()

        if "flow" not in _models:
            try:
                _log(f"  Loading flow model  [{flow_backbone}]…")

                if flow_backbone in ("WAFT-twins", "WAFT-dav2"):
                    if flow_backbone == "WAFT-dav2":
                        da_ckpt = "depth-anything-ckpts/depth_anything_v2_vits.pth"
                        if not os.path.isfile(da_ckpt):
                            _log(f"  ✗ WAFT-dav2 requires depth_anything_v2_vits.pth")
                            _log(f"    Download: {_DAV2_DOWNLOAD_URL}")
                            _log(f"    Save to: {da_ckpt}")
                            raise FileNotFoundError(
                                f"Missing checkpoint: {da_ckpt}\n"
                                f"Download from {_DAV2_DOWNLOAD_URL}"
                            )
                    ckpt = load_file_from_url(
                        url=os.path.join(PRETRAIN_URL, "waft-downstream.pth"),
                        model_dir=WEIGHTS_DIR, progress=False, file_name=None,
                    )
                    _models["flow"] = WAFT_bi(ckpt, device, backbone=flow_backbone,
                                              fp16=fp16_waft)
                    _log(f"  ✓ {flow_backbone}{'  [FP16]' if fp16_waft else ''}  [{_vram_str()}]")

                elif flow_backbone == "RAFT":
                    raft_ckpt = os.path.join(WEIGHTS_DIR, "raft-things.pth")
                    if not os.path.isfile(raft_ckpt):
                        _log(f"  ✗ RAFT requires weights/raft-things.pth")
                        _log("    Download: https://drive.google.com/file/d/1MqDajR89k-xLV0HIrmJ0k-n8ZpG6_suM")
                        raise FileNotFoundError(f"Missing checkpoint: {raft_ckpt}")
                    _models["flow"] = WAFT_bi(raft_ckpt, device, backbone="RAFT",
                                              fp16=fp16_waft)
                    _log(f"  ✓ RAFT{'  [FP16]' if fp16_waft else ''}  [{_vram_str()}]")

                elif flow_backbone == "SEA-RAFT":
                    searaft_ckpt = os.path.join(WEIGHTS_DIR, "sea-raft-M.pth")
                    if not os.path.isfile(searaft_ckpt):
                        _log(f"  ✗ SEA-RAFT requires weights/sea-raft-M.pth")
                        _log("    Clone: https://github.com/princeton-vl/SEA-RAFT")
                        raise FileNotFoundError(f"Missing checkpoint: {searaft_ckpt}")
                    ckpt_size = os.path.getsize(searaft_ckpt)
                    if ckpt_size <= 100:
                        _log(f"  ⚠ sea-raft-M.pth is only {ckpt_size} bytes — likely a bad download")
                    _models["flow"] = WAFT_bi(searaft_ckpt, device, backbone="SEA-RAFT",
                                              fp16=fp16_waft)
                    _log(f"  ✓ SEA-RAFT{'  [FP16]' if fp16_waft else ''}  [{_vram_str()}]")

                _models["flow_backbone"] = flow_backbone
                _models["flow_fp16"] = fp16_waft

            except Exception as exc:
                _log(f"  ✗ {flow_backbone} flow failed — {exc}")
                _log("    (zero flow fallback; quality degraded)")
                errors.append(("flow", str(exc)))

        if "flow_complete" not in _models:
            try:
                _log("  Loading RecurrentFlowCompleteNet…")
                ckpt = load_file_from_url(
                    url=os.path.join(PRETRAIN_URL, "recurrent_flow_completion.pth"),
                    model_dir=WEIGHTS_DIR, progress=False, file_name=None,
                )
                m = RecurrentFlowCompleteNet(ckpt)
                for p in m.parameters():
                    p.requires_grad = False
                m.to(device).eval()
                _models["flow_complete"] = m
                _log(f"  ✓ FlowComplete  [{_vram_str()}]")
            except Exception as exc:
                _log(f"  ✗ FlowComplete failed — {exc}")
                errors.append(("flow_complete", str(exc)))

        if "inpaint" not in _models:
            try:
                _log("  Loading ProPainter inpaint model…")
                ckpt = load_file_from_url(
                    url=os.path.join(PRETRAIN_URL, "ProPainter.pth"),
                    model_dir=WEIGHTS_DIR, progress=False, file_name=None,
                )
                m = InpaintGenerator(model_path=ckpt).to(device)
                m.eval()
                _models["inpaint"] = m
                _log(f"  ✓ ProPainter  [{_vram_str()}]")
            except Exception as exc:
                _log(f"  ✗ ProPainter failed — {exc}")
                errors.append(("inpaint", str(exc)))

    return errors


def _patch_turboquant(enabled: bool, bits: int = 3):
    model = _models.get("inpaint")
    if model is None:
        return
    device = _get_device()
    for mod in model.modules():
        if not isinstance(mod, SparseWindowAttention):
            continue
        mod.use_turboquant = enabled
        if not enabled:
            continue
        c_head = mod.key.out_features // mod.n_head
        if not hasattr(mod, "tq_cache"):
            from model.modules.turboquant_kv import BidirectionalTQKVCache
            mod.tq_cache = BidirectionalTQKVCache(c_head=c_head, bits=bits).to(device)
        elif mod.tq_cache.compressor.bits != bits:
            from model.modules.turboquant_kv import BidirectionalTQKVCache
            mod.tq_cache = BidirectionalTQKVCache(c_head=c_head, bits=bits).to(device)


# ── video / mask utilities ────────────────────────────────────────────────────

def _read_video(path: str):
    """Return (list[PIL.Image], fps)."""
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames, fps


def _masks_from_video(path: str, video_length: int, size: tuple,
                      flow_dilates: int = 4, mask_dilates: int = 4):
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    raw = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray    = frame[:, :, 0]
        resized = cv2.resize(gray, size, interpolation=cv2.INTER_NEAREST)
        raw.append((resized > 127).astype(np.uint8))
    cap.release()

    if not raw:
        return None, None

    while len(raw) < video_length:
        raw.append(raw[-1].copy())
    raw = raw[:video_length]

    batch = np.stack(raw, axis=0)

    def _dilate(arr, iters):
        return binary_dilation(arr, iterations=iters).astype(np.uint8) if iters > 0 else arr

    return (
        [Image.fromarray(f * 255, "L") for f in _dilate(batch, flow_dilates)],
        [Image.fromarray(m * 255, "L") for m in _dilate(batch, mask_dilates)],
    )


_FC_TILE_H = 1080
_FC_TILE_W = 1920


def _tile_flow_complete(flow_complete, gt_flows_bi, fmasks_t, device, overlap=64):
    """
    Run flow_complete.forward_bidirect_flow + combine_flow on overlapping 2×2 spatial tiles.
    Used when the full-resolution Conv3d workspace exceeds available VRAM (typically 4K input).

    gt_flows_bi : (fwd [b, t-1, 2, h, w],  bwd [b, t-1, 2, h, w])
    fmasks_t    : [b, t, 1, h, w]
    Returns     : (pred_fwd, pred_bwd) each [b, t-1, 2, h, w], same dtype as input
    """
    b, t_minus_1, _, h, w = gt_flows_bi[0].shape
    th = (h + 1) // 2
    tw = (w + 1) // 2

    tile_coords = [
        (0,      0     ),
        (0,      w - tw),
        (h - th, 0     ),
        (h - th, w - tw),
    ]

    in_dtype   = gt_flows_bi[0].dtype
    fwd_acc    = torch.zeros(b, t_minus_1, 2, h, w, device=device, dtype=torch.float32)
    bwd_acc    = torch.zeros_like(fwd_acc)
    weight_acc = torch.zeros(b, 1, 1, h, w, device=device, dtype=torch.float32)

    for (r, c) in tile_coords:
        r2, c2 = r + th, c + tw

        gf_fwd = gt_flows_bi[0][:, :, :, r:r2, c:c2]
        gf_bwd = gt_flows_bi[1][:, :, :, r:r2, c:c2]
        fm_t   = fmasks_t[:, :, :, r:r2, c:c2]

        with torch.backends.cudnn.flags(benchmark=False, deterministic=True):
            pred_bi, _ = flow_complete.forward_bidirect_flow((gf_fwd, gf_bwd), fm_t)
            pred_bi    = flow_complete.combine_flow((gf_fwd, gf_bwd), pred_bi, fm_t)

        # linear blend weight — ramps in overlap zones
        wy = torch.ones(th, device=device, dtype=torch.float32)
        wx = torch.ones(tw, device=device, dtype=torch.float32)
        if r > 0:   wy[:overlap]  = torch.linspace(0, 1, overlap, device=device)
        if r2 < h:  wy[-overlap:] = torch.linspace(1, 0, overlap, device=device)
        if c > 0:   wx[:overlap]  = torch.linspace(0, 1, overlap, device=device)
        if c2 < w:  wx[-overlap:] = torch.linspace(1, 0, overlap, device=device)
        wt = (wy.view(1, 1, 1, th, 1) * wx.view(1, 1, 1, 1, tw))  # [1,1,1,th,tw]

        fwd_acc[:, :, :, r:r2, c:c2]    += pred_bi[0].float() * wt
        bwd_acc[:, :, :, r:r2, c:c2]    += pred_bi[1].float() * wt
        weight_acc[:, :, :, r:r2, c:c2] += wt

        del gf_fwd, gf_bwd, fm_t, pred_bi, wt, wy, wx
        torch.cuda.empty_cache()

    weight_acc = weight_acc.clamp(min=1e-6)
    return (
        (fwd_acc / weight_acc).to(in_dtype),
        (bwd_acc / weight_acc).to(in_dtype),
    )


def _write_comparison(left_frames, right_frames, fps: float, out_path: str):
    with imageio.get_writer(out_path, fps=fps, quality=7, macro_block_size=1) as w:
        for lf, rf in zip(left_frames, right_frames):
            l = np.array(lf) if not isinstance(lf, np.ndarray) else lf
            r = np.array(rf) if not isinstance(rf, np.ndarray) else rf
            w.append_data(np.concatenate([l, r], axis=1))


# ── inference ─────────────────────────────────────────────────────────────────

def run_inpainting(
    video_path: str,
    mask_path: str,
    mode: str,
    run_label: str,
    neighbor_length: int,
    ref_stride: int,
    subvideo_length: int,
    fp16: bool,
    use_tq: bool,
    tq_bits: int,
    mask_dilation: int,
    flow_backbone: str,
    fp16_waft: bool,
    num_chunks: int,
    scale_h: float,
    scale_w: float,
    resize_ratio: float,
    res_width: int,
    res_height: int,
    save_fps: str,
    save_frames: bool,
    raft_iters: int,
    log_fn,         # callable(str)
    progress_fn,    # callable(float, str)
    vram_sample_fn, # callable(float, float) — (elapsed_s, vram_gb)
    done_fn,        # callable(str|None, str|None, dict)
):
    """
    Blocking — call from a background thread.
    done_fn receives (out_path, cmp_path, stats_dict).
    stats_dict keys: total_time, peak_vram, chunk_times, run_label.
    """
    _log_file = None

    def _log(msg):
        log_fn(msg)
        if _log_file is not None:
            _log_file.write(msg + "\n")

    def _prog(val, desc=""):
        progress_fn(val, desc)

    def _vlog(label):
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1e9
            resv  = torch.cuda.memory_reserved()  / 1e9
            _log(f"    [VRAM] {label}: alloc={alloc:.2f}GB  reserved={resv:.2f}GB")

    _log(f"  Mode: {mode}  Settings: neighbor={neighbor_length} ref_stride={ref_stride} "
         f"subvideo={subvideo_length} dilation={mask_dilation} "
         f"backbone={flow_backbone} fp16={fp16} fp16_waft={fp16_waft} "
         f"tq={use_tq} tq_bits={tq_bits} iters={raft_iters}")

    try:
        if not video_path or not os.path.isfile(video_path):
            _log("❌  Video file not found.")
            done_fn(None, None, {})
            return

        if mode == "video_inpainting":
            if not mask_path or not os.path.isfile(mask_path):
                _log("❌  Mask file not found.")
                done_fn(None, None, {})
                return

        run_label  = run_label.strip() or "run"
        safe_label = run_label.replace(" ", "_")
        video_name = Path(video_path).stem
        _ts        = time.strftime("%Y%m%d_%H%M%S")
        os.makedirs(RESULTS_DIR, exist_ok=True)
        _log_file  = open(
            os.path.join(RESULTS_DIR, f"{video_name}_{safe_label}_{_ts}.log"),
            "w", buffering=1, encoding="utf-8",
        )
        _log(f"  Log: {_log_file.name}")

        mask_is_video = (
            mode == "video_inpainting"
            and bool(mask_path)
            and Path(mask_path).suffix.lower() in _VIDEO_EXTS
        )
        peak_vram      = 0.0
        chunk_times: list[float] = []

        # ── load models ───────────────────────────────────────────────────────
        _log("── Loading models ─────────────────────────────────────")
        _prog(0.0, "Loading models…")
        load_models(log_fn=_log, flow_backbone=flow_backbone, fp16_waft=fp16_waft)

        if "flow_complete" not in _models or "inpaint" not in _models:
            _log("❌  Critical models failed — cannot continue.")
            done_fn(None, None, {})
            return

        _patch_turboquant(bool(use_tq), int(tq_bits))
        _log(f"TurboQuant: {'ON  bits={}'.format(int(tq_bits)) if use_tq else 'OFF'}")

        device   = _get_device()
        use_half = bool(fp16) and device.type == "cuda"

        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = False  # keep speed, just not benchmark mode
            # expandable_segments lets the allocator grow/shrink segments rather than
            # holding fragmented blocks — reduces the reserved-but-unallocated gap
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        # ── device info ───────────────────────────────────────────────────────
        _log("── Compute device ─────────────────────────────────────")
        if device.type == 'cuda':
            _log(f"  Device: {torch.cuda.get_device_name(0)}")
            _log(f"  CUDA: {torch.version.cuda}  |  PyTorch: {torch.__version__}")
            total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            _log(f"  VRAM total: {total_vram:.1f} GB")
        else:
            _log("  ⚠ Device: CPU — CUDA not detected")
            _log("  Check CUDA DLL PATH — see Context.md Build Environment section")

        # ── read video ────────────────────────────────────────────────────────
        _log("── Reading video ──────────────────────────────────────")
        _prog(0.05, "Reading video…")
        try:
            frames_pil, fps = _read_video(video_path)
        except Exception:
            _log(f"❌  {traceback.format_exc()}")
            done_fn(None, None, {})
            return

        if not frames_pil:
            _log("❌  Video contains no readable frames.")
            done_fn(None, None, {})
            return

        video_name = Path(video_path).stem

        # ── resize (resolution_expansion overrides target size) ───────────────
        if mode == "resolution_expansion":
            orig_w, orig_h = frames_pil[0].size
            if int(res_width) > 0 and int(res_height) > 0:
                target_size = (int(res_width), int(res_height))
                _log(f"  Resolution expansion: explicit {target_size[0]}×{target_size[1]}")
            else:
                target_size = (int(float(resize_ratio) * orig_w),
                               int(float(resize_ratio) * orig_h))
                _log(f"  Resolution expansion: {float(resize_ratio):.1f}× → "
                     f"{target_size[0]}×{target_size[1]}")
            frames_pil, size, out_size = resize_frames(frames_pil, target_size)
        else:
            frames_pil, size, out_size = resize_frames(frames_pil)
        w, h         = size
        video_length = len(frames_pil)
        if num_chunks > 0:
            subvideo_length = max(20, video_length // num_chunks)
            _log(f"  Auto subvideo_length: {video_length} frames ÷ {num_chunks} chunks = {subvideo_length}")
        frames_inp = [np.array(f).astype(np.uint8) for f in frames_pil]
        _log(f"  {video_length} frames | proc {w}×{h} | out {out_size[0]}×{out_size[1]} | {fps:.2f} fps")

        # ── build masks / extrapolate ─────────────────────────────────────────
        _log("── Building masks ─────────────────────────────────────")
        _prog(0.08, "Building masks…")
        try:
            if mode == "video_outpainting":
                frames_pil, flow_masks, masks_dilated, size = extrapolation(
                    frames_pil, (float(scale_h), float(scale_w))
                )
                w, h      = size
                out_size  = size
                frames_inp = [np.array(f).astype(np.uint8) for f in frames_pil]
                video_length = len(frames_pil)
                _log(f"  Outpainting: scale ({float(scale_h):.2f}, {float(scale_w):.2f}) → {w}×{h}")
            elif mode == "resolution_expansion":
                mask_arr = np.ones((h, w), dtype=np.uint8) * 255
                _mask_pil = Image.fromarray(mask_arr, "L")
                flow_masks    = [_mask_pil] * video_length
                masks_dilated = [_mask_pil] * video_length
                _log(f"  Resolution expansion: full-frame mask at {w}×{h}")
            else:  # video_inpainting
                if mask_is_video:
                    flow_masks, masks_dilated = _masks_from_video(
                        mask_path, video_length, size,
                        flow_dilates=int(mask_dilation),
                        mask_dilates=int(mask_dilation),
                    )
                    if flow_masks is None:
                        _log("❌  Could not read mask video frames.")
                        done_fn(None, None, {})
                        return
                    _log(f"  Video mask: {len(flow_masks)} frames | dilation {mask_dilation}px")
                else:
                    tmp_dir  = tempfile.mkdtemp(prefix="propainter_")
                    mask_pil = Image.open(mask_path).convert("L").resize(size, Image.NEAREST)
                    mask_tmp = os.path.join(tmp_dir, "mask.png")
                    mask_pil.save(mask_tmp)
                    flow_masks, masks_dilated = read_mask(
                        mask_tmp, video_length, size,
                        flow_mask_dilates=int(mask_dilation),
                        mask_dilates=int(mask_dilation),
                    )
                    _log(f"  Image mask | dilation {mask_dilation}px | {len(flow_masks)} frames ready")
        except Exception:
            _log(f"❌  Mask processing failed:\n{traceback.format_exc()}")
            done_fn(None, None, {})
            return

        # ── masked overlay frames for comparison video ────────────────────────
        masked_for_save = []
        for fp2, mdil_pil in zip(frames_pil, masks_dilated):
            frame_np = np.array(fp2)
            msk_np   = np.expand_dims(np.array(mdil_pil), 2).repeat(3, axis=2) / 255.0
            green    = np.zeros((h, w, 3), dtype=np.float32)
            green[:, :, 1] = 255.0
            fuse     = 0.4 * frame_np + 0.6 * green
            masked_for_save.append((msk_np * fuse + (1.0 - msk_np) * frame_np).astype(np.uint8))

        # ── precision cast ────────────────────────────────────────────────────
        flow_model    = _models.get("flow")
        flow_complete = _models["flow_complete"]
        inpaint_model = _models["inpaint"]

        if use_half:
            flow_complete = flow_complete.half()
            inpaint_model = inpaint_model.half()
            _log("FP16 enabled")
        else:
            flow_complete = flow_complete.float()
            inpaint_model = inpaint_model.float()

        # ── chunk inference loop ──────────────────────────────────────────────
        _log("── Running inference ──────────────────────────────────")

        overlap      = 5
        chunk_size   = int(subvideo_length)
        comp_frames  = [None] * video_length
        comp_weights = [0.0]  * video_length

        chunk_starts = list(range(0, video_length, chunk_size - overlap))
        n_chunks     = len(chunk_starts)
        wall_t0      = time.perf_counter()

        for ci, chunk_start in enumerate(chunk_starts):
            chunk_end = min(chunk_start + chunk_size, video_length)
            chunk_len = chunk_end - chunk_start
            if chunk_len == 0:
                break

            c_frames = frames_pil[chunk_start:chunk_end]
            c_fmasks = flow_masks[chunk_start:chunk_end]
            c_mdil   = masks_dilated[chunk_start:chunk_end]
            c_inp    = frames_inp[chunk_start:chunk_end]

            _prog(
                0.1 + 0.85 * ci / n_chunks,
                f"Chunk {ci+1}/{n_chunks}  frames {chunk_start}–{chunk_end-1}",
            )
            _log(f"  Chunk {ci+1}/{n_chunks}  frames {chunk_start}–{chunk_end-1}")

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            if flow_model is not None:
                flow_model.to(device)
                _vlog("WAFT→GPU")

            frames_t = to_tensors()(c_frames).unsqueeze(0) * 2 - 1
            fmasks_t = to_tensors()(c_fmasks).unsqueeze(0)
            mdil_t   = to_tensors()(c_mdil).unsqueeze(0)
            frames_t = frames_t.to(device)
            fmasks_t = fmasks_t.to(device)
            mdil_t   = mdil_t.to(device)
            _vlog("chunk tensors→GPU")

            chunk_t0 = time.perf_counter()

            with torch.inference_mode():
                if   frames_t.size(-1) <= 640:  scl = 12
                elif frames_t.size(-1) <= 720:  scl = 8
                elif frames_t.size(-1) <= 1280: scl = 4
                else:                           scl = 2

                if flow_model is not None:
                    if frames_t.size(1) > scl:
                        fwd_list, bwd_list = [], []
                        for f in range(0, chunk_len, scl):
                            ef = min(chunk_len, f + scl)
                            sl = frames_t[:, f:ef] if f == 0 else frames_t[:, f - 1:ef]
                            ff, fb = flow_model(sl, iters=raft_iters)
                            fwd_list.append(ff.cpu())
                            bwd_list.append(fb.cpu())
                            del ff, fb
                            torch.cuda.empty_cache()
                            _vlog(f"  WAFT sub {f}–{ef} done")
                        gt_flows_bi = (
                            torch.cat(fwd_list, dim=1).to(device),
                            torch.cat(bwd_list, dim=1).to(device),
                        )
                        del fwd_list, bwd_list
                        torch.cuda.empty_cache()
                    else:
                        gt_flows_bi = flow_model(frames_t, iters=raft_iters)
                        torch.cuda.empty_cache()
                else:
                    _log("    ⚠ WAFT not loaded — zero flow fallback")
                    B, T, C, H, W = frames_t.shape
                    gt_flows_bi = (
                        torch.zeros(B, T - 1, 2, H, W, device=device),
                        torch.zeros(B, T - 1, 2, H, W, device=device),
                    )

                if flow_model is not None:
                    flow_model.cpu()
                    torch.cuda.empty_cache()
                    _log(f"  WAFT offloaded to CPU  [{_vram_str()}]")
                    _vlog("WAFT→CPU offloaded")

                if use_half:
                    frames_t    = frames_t.half()
                    fmasks_t    = fmasks_t.half()
                    mdil_t      = mdil_t.half()
                    gt_flows_bi = (gt_flows_bi[0].half(), gt_flows_bi[1].half())

                if h > _FC_TILE_H or w > _FC_TILE_W:
                    pred_flows_bi = _tile_flow_complete(
                        flow_complete, gt_flows_bi, fmasks_t, device
                    )
                else:
                    with torch.backends.cudnn.flags(benchmark=False, deterministic=True):
                        pred_flows_bi, _ = flow_complete.forward_bidirect_flow(gt_flows_bi, fmasks_t)
                        pred_flows_bi    = flow_complete.combine_flow(gt_flows_bi, pred_flows_bi, fmasks_t)
                torch.cuda.empty_cache()
                _vlog("flow_complete done")

                masked_f = frames_t * (1 - mdil_t)
                b, t, _, _, _ = mdil_t.size()
                prop_imgs, upd_masks = inpaint_model.img_propagation(
                    masked_f, pred_flows_bi, mdil_t, "nearest"
                )
                upd_frames = frames_t * (1 - mdil_t) + prop_imgs.view(b, t, 3, h, w) * mdil_t
                upd_masks  = upd_masks.view(b, t, 1, h, w)
                torch.cuda.empty_cache()
                _vlog("img_propagation done")

            neighbor_stride = int(neighbor_length) // 2
            ref_num = chunk_size // int(ref_stride) if chunk_len > chunk_size // 2 else -1

            for f in range(0, chunk_len, neighbor_stride):
                nb_ids  = [
                    i for i in range(
                        max(0, f - neighbor_stride),
                        min(chunk_len, f + neighbor_stride + 1),
                    )
                ]
                ref_ids = get_ref_index(f, nb_ids, chunk_len, int(ref_stride), ref_num)

                sel_imgs     = upd_frames[:, nb_ids + ref_ids]
                sel_masks    = mdil_t[:, nb_ids + ref_ids]
                sel_upd_mask = upd_masks[:, nb_ids + ref_ids]
                sel_flows    = (
                    pred_flows_bi[0][:, nb_ids[:-1]],
                    pred_flows_bi[1][:, nb_ids[:-1]],
                )

                with torch.inference_mode():
                    _vlog(f"  inpaint f={f} pre-forward")
                    l_t  = len(nb_ids)
                    pred = inpaint_model(sel_imgs, sel_flows, sel_masks, sel_upd_mask, l_t)
                    pred = pred.view(-1, 3, h, w)
                    pred = (pred + 1) / 2
                    pred_np   = pred.cpu().permute(0, 2, 3, 1).numpy() * 255
                    bin_masks = mdil_t[0, nb_ids].cpu().permute(0, 2, 3, 1).numpy().astype(np.uint8)
                    del pred
                    _vlog(f"  inpaint f={f} post-del pred")

                    for local_i, chunk_i in enumerate(nb_ids):
                        global_i = chunk_start + chunk_i
                        img_out  = (
                            pred_np[local_i].astype(np.uint8) * bin_masks[local_i]
                            + c_inp[chunk_i] * (1 - bin_masks[local_i])
                        )
                        weight = 1.0
                        if overlap > 0:
                            if chunk_i < overlap and chunk_start > 0:
                                weight = chunk_i / overlap
                            elif chunk_i >= chunk_len - overlap and chunk_end < video_length:
                                weight = (chunk_len - chunk_i) / overlap
                        if comp_frames[global_i] is None:
                            comp_frames[global_i]  = img_out.astype(np.float32) * weight
                            comp_weights[global_i] = weight
                        else:
                            comp_frames[global_i]  += img_out.astype(np.float32) * weight
                            comp_weights[global_i] += weight

                del sel_imgs, sel_masks, sel_upd_mask, sel_flows, pred_np, bin_masks
                torch.cuda.empty_cache()
                _vlog(f"  inpaint f={f} post-cleanup")

            elapsed   = time.perf_counter() - chunk_t0
            chunk_times.append(elapsed)
            vram_now  = _vram_alloc_gb()
            vram_peak_chunk = (
                torch.cuda.max_memory_allocated() / 1e9
                if torch.cuda.is_available() else 0.0
            )
            peak_vram = max(peak_vram, vram_peak_chunk)
            vram_sample_fn(time.perf_counter() - wall_t0, vram_now)

            _log(f"    ✓ {elapsed:.1f}s | peak VRAM {vram_peak_chunk:.2f} GB")

            del frames_t, fmasks_t, mdil_t, gt_flows_bi, pred_flows_bi
            del masked_f, prop_imgs, upd_frames, upd_masks
            torch.cuda.empty_cache()
            _vlog("chunk teardown complete")

        # ── post-loop CUDA flush ─────────────────────────────────────────────
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        _log(f"  Post-loop VRAM: [{_vram_str()}]")

        # ── normalise blended frames ──────────────────────────────────────────
        for i in range(video_length):
            if comp_frames[i] is None:
                comp_frames[i] = frames_inp[i]
            elif comp_weights[i] > 0:
                comp_frames[i] = (comp_frames[i] / comp_weights[i]).astype(np.uint8)

        total_time = time.perf_counter() - wall_t0
        _log(f"── Complete  {total_time:.1f}s total  [{_vram_str()}]")

        # ── write output videos ───────────────────────────────────────────────
        _prog(0.97, "Writing output videos…")
        os.makedirs(RESULTS_DIR, exist_ok=True)

        # FPS override
        out_fps = fps
        try:
            _sfps = float(save_fps) if save_fps else 0.0
            if _sfps > 0:
                out_fps = _sfps
                _log(f"  FPS override: {out_fps}")
        except (ValueError, TypeError):
            pass

        out_mp4 = os.path.join(RESULTS_DIR, f"{video_name}_{safe_label}_inpainted.mp4")
        cmp_mp4 = os.path.join(RESULTS_DIR, f"{video_name}_{safe_label}_comparison.mp4")

        comp_out = [cv2.resize(f, out_size) for f in comp_frames]
        del comp_frames  # free normalised float32 list before masked list is built

        masked_out = [cv2.resize(f, out_size) for f in masked_for_save]
        del masked_for_save  # free masked frames before writing

        _log(f"  Writing {out_mp4}")
        imageio.mimwrite(out_mp4, comp_out, fps=out_fps, quality=7, macro_block_size=1)
        _log(f"  Writing {cmp_mp4}")
        _write_comparison(masked_out, comp_out, out_fps, cmp_mp4)

        out_mb = os.path.getsize(out_mp4) / 1e6 if os.path.exists(out_mp4) else 0.0
        cmp_mb = os.path.getsize(cmp_mp4) / 1e6 if os.path.exists(cmp_mp4) else 0.0
        _log(f"  ✓ inpainted  {out_mb:.2f} MB")
        _log(f"  ✓ comparison {cmp_mb:.2f} MB")

        del comp_out, masked_out  # both videos written; free before PNG export

        # ── save PNG frame sequence ───────────────────────────────────────────
        if save_frames:
            # Re-read comp_out from the written MP4 to avoid holding a second copy
            comp_out_png = [cv2.resize(f, out_size) for f in
                            [cv2.cvtColor(np.array(fr), cv2.COLOR_RGB2BGR)
                             for fr in imageio.mimread(out_mp4)]]
            frames_dir = os.path.join(RESULTS_DIR, f"{video_name}_{safe_label}_frames")
            os.makedirs(frames_dir, exist_ok=True)
            _prog(0.98, "Writing PNG frames…")
            for idx, frame in enumerate(comp_out_png):
                cv2.imwrite(os.path.join(frames_dir, f"{idx:05d}.png"), frame)
            del comp_out_png
            _log(f"  ✓ frames → {frames_dir}/")

        _prog(1.0, "Done")

        stats = {
            "run_label":      run_label,
            "total_time":     total_time,
            "peak_vram":      peak_vram,
            "chunk_times":    chunk_times,
            "avg_chunk_time": sum(chunk_times) / len(chunk_times) if chunk_times else 0.0,
        }
        done_fn(out_mp4, cmp_mp4, stats)

    except Exception:
        _log("── INFERENCE FAILED ───────────────────────────────────")
        _log(traceback.format_exc())
        _log("── UI reset — you can change settings and try again ──")
        done_fn(None, None, {})
    finally:
        if _log_file is not None:
            _log_file.close()


# ── Tooltip helper ───────────────────────────────────────────────────────────

class _Tooltip:
    """Show a tooltip label when the mouse hovers over a widget."""
    def __init__(self, widget, text):
        self._widget = widget
        self._text   = text
        self._tip    = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, event=None):
        x = self._widget.winfo_rootx() + 20
        y = self._widget.winfo_rooty() + self._widget.winfo_height() + 4
        self._tip = tk.Toplevel(self._widget)
        self._tip.wm_overrideredirect(True)
        self._tip.wm_geometry(f"+{x}+{y}")
        ttk.Label(self._tip, text=self._text, relief="solid",
                  padding=4, wraplength=320).pack()

    def _hide(self, event=None):
        if self._tip:
            self._tip.destroy()
            self._tip = None


# ── Tkinter application ───────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ProPainter — Phase 1 Benchmark Harness")
        self.resizable(True, True)
        self.minsize(920, 620)

        self._running        = False
        self._log_queue: list[str] = []
        self._log_lock       = threading.Lock()

        # Benchmark data: list of dicts with run_label, total_time, peak_vram,
        # avg_chunk_time, and time_series [(elapsed_s, vram_gb), ...]
        self._bench_data: list[dict] = []
        # Currently-accumulating VRAM samples for the active run
        self._active_samples: list[tuple[float, float]] = []
        self._vram_after_id = None

        self._build_ui()
        self._schedule_vram()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        nb = ttk.Notebook(self)
        nb.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        self._tab_run  = ttk.Frame(nb)
        self._tab_bench = ttk.Frame(nb)
        self._tab_cmp  = ttk.Frame(nb)

        nb.add(self._tab_run,   text="  Run  ")
        nb.add(self._tab_bench, text="  Benchmark  ")
        nb.add(self._tab_cmp,   text="  Compare  ")

        self._build_run_tab(self._tab_run)
        self._build_bench_tab(self._tab_bench)
        self._build_cmp_tab(self._tab_cmp)

    # ── Tab 1: Run ────────────────────────────────────────────────────────────

    def _build_run_tab(self, parent):
        parent.columnconfigure(0, weight=0, minsize=360)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(0, weight=1)

        left  = ttk.Frame(parent, padding=8)
        right = ttk.Frame(parent, padding=(0, 8, 8, 8))
        left .grid(row=0, column=0, sticky="nsew")
        right.grid(row=0, column=1, sticky="nsew")

        self._build_run_left(left)
        self._build_run_right(right)

    def _build_run_left(self, p):
        p.columnconfigure(1, weight=1)
        row = 0

        # ── Mode selector ─────────────────────────────────────────────────────
        ttk.Label(p, text="Mode:").grid(row=row, column=0, sticky="w", pady=2)
        self._mode_var = tk.StringVar(value="video_inpainting")
        _mode_combo = ttk.Combobox(
            p, textvariable=self._mode_var, state="readonly", width=24,
            values=["video_inpainting", "video_outpainting", "resolution_expansion"],
        )
        _mode_combo.grid(row=row, column=1, columnspan=2, sticky="w", padx=(4, 0), pady=2)
        _Tooltip(_mode_combo,
                 "Video Inpainting: remove objects with a mask. "
                 "Video Outpainting: expand the canvas beyond original borders. "
                 "Resolution Expansion: upscale using temporal propagation.")
        row += 1

        ttk.Separator(p, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=6)
        row += 1

        # ── Video path (all modes) ─────────────────────────────────────────────
        ttk.Label(p, text="Video path:").grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(4, 0))
        row += 1
        self._video_var = tk.StringVar()
        ttk.Entry(p, textvariable=self._video_var).grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=2)
        ttk.Button(p, text="Browse…", command=self._browse_video, width=8).grid(
            row=row, column=2, padx=(4, 0), pady=2)
        row += 1

        # ── Mode-specific section (three frames, same row, only one visible) ──
        MODE_FRAME_ROW = row
        row += 1

        # --- inpainting frame: mask path + dilation ---
        fi = ttk.Frame(p)
        fi.columnconfigure(1, weight=1)
        ttk.Label(fi, text="Mask path  (image or video):").grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(4, 0))
        self._mask_var = tk.StringVar()
        ttk.Entry(fi, textvariable=self._mask_var).grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=2)
        ttk.Button(fi, text="Browse…", command=self._browse_mask, width=8).grid(
            row=1, column=2, padx=(4, 0), pady=2)
        self._dil_var = tk.IntVar(value=4)
        ttk.Label(fi, text="Mask dilation (px):").grid(row=2, column=0, sticky="w", pady=2)
        _dil_sb = ttk.Spinbox(fi, from_=0, to=20, textvariable=self._dil_var, width=6)
        _dil_sb.grid(row=2, column=1, sticky="w", padx=(4, 0), pady=2)
        _Tooltip(_dil_sb,
                 "Expands the mask outward by this many pixels. Helps cover hard edges "
                 "and compression artifacts around the masked region. Default 4.")
        self._frame_inpaint = fi

        # --- outpainting frame: scale_h + scale_w ---
        fo = ttk.Frame(p)
        fo.columnconfigure(1, weight=1)
        self._scale_h_var = tk.DoubleVar(value=1.0)
        ttk.Label(fo, text="Scale H:").grid(row=0, column=0, sticky="w", pady=2)
        _sh_sb = ttk.Spinbox(fo, from_=1.0, to=4.0, increment=0.05, format="%.2f",
                              textvariable=self._scale_h_var, width=7)
        _sh_sb.grid(row=0, column=1, sticky="w", padx=(4, 0), pady=2)
        _Tooltip(_sh_sb,
                 "Vertical scale multiplier for outpainting. 1.5 = 50% taller canvas. "
                 "Content is generated by ProPainter to fill the new area.")
        self._scale_w_var = tk.DoubleVar(value=1.5)
        ttk.Label(fo, text="Scale W:").grid(row=1, column=0, sticky="w", pady=2)
        _sw_sb = ttk.Spinbox(fo, from_=1.0, to=4.0, increment=0.05, format="%.2f",
                              textvariable=self._scale_w_var, width=7)
        _sw_sb.grid(row=1, column=1, sticky="w", padx=(4, 0), pady=2)
        _Tooltip(_sw_sb,
                 "Horizontal scale multiplier for outpainting. "
                 "1.78 on a 9:16 video gives 16:9 output.")
        self._frame_outpaint = fo

        # --- resolution expansion frame: resize_ratio + width + height ---
        fe = ttk.Frame(p)
        fe.columnconfigure(1, weight=1)
        self._resize_ratio_var = tk.DoubleVar(value=2.0)
        ttk.Label(fe, text="Resize ratio:").grid(row=0, column=0, sticky="w", pady=2)
        _rr_sb = ttk.Spinbox(fe, from_=0.5, to=8.0, increment=0.5, format="%.1f",
                              textvariable=self._resize_ratio_var, width=6)
        _rr_sb.grid(row=0, column=1, sticky="w", padx=(4, 0), pady=2)
        _Tooltip(_rr_sb,
                 "Uniform scale multiplier for resolution expansion. "
                 "2.0 = 2× upscale. Overridden if explicit width/height are set.")
        self._res_w_var = tk.IntVar(value=0)
        ttk.Label(fe, text="Width (0 = auto):").grid(row=1, column=0, sticky="w", pady=2)
        _rw_sb = ttk.Spinbox(fe, from_=0, to=7680, increment=8,
                              textvariable=self._res_w_var, width=7)
        _rw_sb.grid(row=1, column=1, sticky="w", padx=(4, 0), pady=2)
        _Tooltip(_rw_sb,
                 "Explicit output resolution for resolution expansion. "
                 "Set both or neither. Overrides resize ratio when set.")
        self._res_h_var = tk.IntVar(value=0)
        ttk.Label(fe, text="Height (0 = auto):").grid(row=2, column=0, sticky="w", pady=2)
        _rh_sb = ttk.Spinbox(fe, from_=0, to=4320, increment=8,
                              textvariable=self._res_h_var, width=7)
        _rh_sb.grid(row=2, column=1, sticky="w", padx=(4, 0), pady=2)
        _Tooltip(_rh_sb,
                 "Explicit output resolution for resolution expansion. "
                 "Set both or neither. Overrides resize ratio when set.")
        self._frame_expand = fe

        # Grid all three at the same row; _on_mode_changed controls visibility
        self._frame_inpaint.grid( row=MODE_FRAME_ROW, column=0, columnspan=3, sticky="ew")
        self._frame_outpaint.grid(row=MODE_FRAME_ROW, column=0, columnspan=3, sticky="ew")
        self._frame_expand.grid(  row=MODE_FRAME_ROW, column=0, columnspan=3, sticky="ew")
        self._mode_var.trace_add("write", self._on_mode_changed)
        self._on_mode_changed()  # set initial visibility

        ttk.Separator(p, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=8)
        row += 1

        # ── Run label ─────────────────────────────────────────────────────────
        ttk.Label(p, text="Run label:").grid(row=row, column=0, sticky="w")
        self._label_var = tk.StringVar(value="baseline")
        _label_entry = ttk.Entry(p, textvariable=self._label_var)
        _label_entry.grid(row=row, column=1, columnspan=2, sticky="ew", padx=(4, 0))
        _Tooltip(_label_entry,
                 "Name for this run. Output files are named {video}_{label}_inpainted.mp4 "
                 "— labels keep runs from overwriting each other for A/B comparison.")
        row += 1

        ttk.Separator(p, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=8)
        row += 1

        def _spin(label, var_attr, default, lo, hi, tip):
            nonlocal row
            var = tk.IntVar(value=default)
            setattr(self, var_attr, var)
            ttk.Label(p, text=label).grid(row=row, column=0, sticky="w", pady=2)
            sb = ttk.Spinbox(p, from_=lo, to=hi, textvariable=var, width=6)
            sb.grid(row=row, column=1, sticky="w", padx=(4, 0), pady=2)
            _Tooltip(sb, tip)
            row += 1

        _spin("Neighbor length:",  "_nb_var", 10, 4, 30,
              "How many frames on each side of the current frame ProPainter uses for "
              "temporal context. Higher = better consistency, more VRAM. Default 10.")
        _spin("Reference stride:", "_rs_var", 10, 2, 20,
              "Step size when selecting reference frames from the full clip. "
              "Lower = denser reference coverage, slower. Default 10.")

        # ── Number of chunks / Subvideo length (linked) ───────────────────────
        self._chunks_var = tk.IntVar(value=0)
        ttk.Label(p, text="Number of chunks:").grid(row=row, column=0, sticky="w", pady=2)
        _chunks_sb = ttk.Spinbox(p, from_=0, to=50, textvariable=self._chunks_var, width=6)
        _chunks_sb.grid(row=row, column=1, sticky="w", padx=(4, 0), pady=2)
        _Tooltip(_chunks_sb,
                 "Set how many chunks to split the video into. ProPainter will "
                 "auto-calculate subvideo length. Set to 0 to use the subvideo "
                 "length setting directly. Fewer chunks = faster but more VRAM per chunk.")
        row += 1

        self._sv_var = tk.IntVar(value=80)
        ttk.Label(p, text="Subvideo length:").grid(row=row, column=0, sticky="w", pady=2)
        self._sv_spin = ttk.Spinbox(p, from_=20, to=200, textvariable=self._sv_var, width=6)
        self._sv_spin.grid(row=row, column=1, sticky="w", padx=(4, 0), pady=2)
        _Tooltip(self._sv_spin,
                 "Frames per chunk. Disabled when Number of chunks is set. "
                 "Reduce manually if you get out-of-memory errors.")
        row += 1

        self._chunks_var.trace_add("write", self._on_chunks_changed)
        self._on_chunks_changed()  # set initial state

        ttk.Separator(p, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=8)
        row += 1

        # ── FP16 options ──────────────────────────────────────────────────────
        self._fp16_var = tk.BooleanVar(value=True)
        _fp16_cb = ttk.Checkbutton(p, text="FP16  (ProPainter half precision)",
                                   variable=self._fp16_var)
        _fp16_cb.grid(row=row, column=0, columnspan=3, sticky="w")
        _Tooltip(_fp16_cb,
                 "Run ProPainter and FlowCompleteNet in half precision. "
                 "Saves ~1-2 GB VRAM with minimal quality loss on RTX 30xx.")
        row += 1

        self._fp16_waft_var = tk.BooleanVar(value=False)
        _fp16_waft_cb = ttk.Checkbutton(p, text="FP16  (WAFT flow model)",
                                        variable=self._fp16_waft_var)
        _fp16_waft_cb.grid(row=row, column=0, columnspan=3, sticky="w")
        _Tooltip(_fp16_waft_cb,
                 "Run the WAFT optical flow model in half precision. "
                 "Additional VRAM saving but may reduce flow accuracy on fine detail.")
        row += 1

        self._tq_var   = tk.BooleanVar(value=False)
        self._bits_var = tk.IntVar(value=3)
        tq_row = ttk.Frame(p)
        tq_row.grid(row=row, column=0, columnspan=3, sticky="w", pady=2)
        _tq_cb = ttk.Checkbutton(tq_row, text="TurboQuant",
                                  variable=self._tq_var,
                                  command=self._on_tq_toggle)
        _tq_cb.pack(side=tk.LEFT)
        _Tooltip(_tq_cb,
                 "Compress ProPainter's KV cache to 2-4 bits using Lloyd-Max quantization. "
                 "Allows larger reference windows without OOM. Off by default — "
                 "A/B test before committing.")
        ttk.Label(tq_row, text="  bits:").pack(side=tk.LEFT)
        self._bits_spin = ttk.Spinbox(tq_row, from_=2, to=4,
                                      textvariable=self._bits_var, width=4)
        self._bits_spin.pack(side=tk.LEFT, padx=(2, 0))
        _Tooltip(self._bits_spin,
                 "Bit depth for KV cache compression. 4-bit is near-lossless. "
                 "3-bit saves more VRAM. 2-bit is experimental.")
        self._on_tq_toggle()
        row += 1

        ttk.Separator(p, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=8)
        row += 1

        # ── Flow backbone + RAFT iterations ───────────────────────────────────
        ttk.Label(p, text="Flow backbone:").grid(row=row, column=0, sticky="w", pady=2)
        self._backbone_var = tk.StringVar(value="WAFT-dav2")
        self._backbone_combo = ttk.Combobox(
            p, textvariable=self._backbone_var, state="readonly", width=14,
            values=["WAFT-twins", "WAFT-dav2", "RAFT", "SEA-RAFT"],
        )
        self._backbone_combo.grid(row=row, column=1, columnspan=2, sticky="w", padx=(4, 0), pady=2)
        _Tooltip(self._backbone_combo,
                 "Which optical flow model to use. WAFT-dav2 is the primary target "
                 "(best quality, depth-aware). RAFT is the original baseline. "
                 "SEA-RAFT requires additional setup.")
        row += 1

        _spin("RAFT iterations:", "_raft_iters_var", 20, 1, 100,
              "Number of refinement iterations for optical flow. Higher = more accurate "
              "flow, slower. Default 20. Reduce to 10-12 for faster previews.")

        ttk.Separator(p, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=8)
        row += 1

        # ── Save options ──────────────────────────────────────────────────────
        ttk.Label(p, text="Save FPS:").grid(row=row, column=0, sticky="w", pady=2)
        self._save_fps_var = tk.StringVar(value="")
        _fps_entry = ttk.Entry(p, textvariable=self._save_fps_var, width=8)
        _fps_entry.grid(row=row, column=1, sticky="w", padx=(4, 0), pady=2)
        _Tooltip(_fps_entry,
                 "Output video frame rate. Leave blank to use source FPS. "
                 "Use 23.976, 24, 25, 29.97, or 30.")
        row += 1

        self._save_frames_var = tk.BooleanVar(value=False)
        _sf_cb = ttk.Checkbutton(p, text="Save frames  (PNG sequence)",
                                  variable=self._save_frames_var)
        _sf_cb.grid(row=row, column=0, columnspan=3, sticky="w")
        _Tooltip(_sf_cb,
                 "Also export each output frame as a PNG sequence to "
                 "results/{label}_frames/. Useful for grading in Resolve.")
        row += 1

        ttk.Separator(p, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=8)
        row += 1

        # ── Run / Open results ────────────────────────────────────────────────
        btn_frame = ttk.Frame(p)
        btn_frame.grid(row=row, column=0, columnspan=3, sticky="ew")
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)

        self._run_btn = ttk.Button(btn_frame, text="▶  Run Inpainting",
                                   command=self._on_run)
        self._run_btn.grid(row=0, column=0, sticky="ew", padx=(0, 4), ipady=4)
        ttk.Button(btn_frame, text="Open results/",
                   command=self._open_results).grid(
            row=0, column=1, sticky="ew", ipady=4)

    def _build_run_right(self, p):
        p.rowconfigure(1, weight=1)
        p.columnconfigure(0, weight=1)

        self._vram_var = tk.StringVar(value=_vram_str())
        ttk.Label(p, textvariable=self._vram_var,
                  relief="sunken", padding=4).grid(
            row=0, column=0, sticky="ew", pady=(0, 4))

        self._log_box = scrolledtext.ScrolledText(
            p, state=tk.DISABLED, wrap=tk.WORD,
            font=("Consolas", 9), height=22,
        )
        self._log_box.grid(row=1, column=0, sticky="nsew")

        self._progress_lbl = tk.StringVar(value="")
        ttk.Label(p, textvariable=self._progress_lbl).grid(
            row=2, column=0, sticky="w", pady=(4, 0))
        self._progress_var = tk.DoubleVar(value=0.0)
        ttk.Progressbar(p, variable=self._progress_var,
                        maximum=1.0).grid(
            row=3, column=0, sticky="ew", pady=(2, 0))

        vram_frame = ttk.Frame(p)
        vram_frame.grid(row=4, column=0, sticky="ew", pady=(6, 0))
        vram_frame.columnconfigure(1, weight=1)
        ttk.Label(vram_frame, text="VRAM").grid(row=0, column=0, sticky="w", padx=(0, 6))
        self._vram_bar = ttk.Progressbar(vram_frame, orient="horizontal",
                                         mode="determinate", maximum=100)
        self._vram_bar.grid(row=0, column=1, sticky="ew")
        self._vram_label = ttk.Label(vram_frame, text="— / — GB", width=22, anchor="e")
        self._vram_label.grid(row=0, column=2, sticky="e", padx=(6, 0))

    # ── Tab 2: Benchmark ──────────────────────────────────────────────────────

    def _build_bench_tab(self, parent):
        parent.rowconfigure(0, weight=3)
        parent.rowconfigure(1, weight=1)
        parent.columnconfigure(0, weight=1)

        chart_frame = ttk.LabelFrame(parent, text="VRAM over time  (GB allocated)",
                                     padding=4)
        chart_frame.grid(row=0, column=0, sticky="nsew", padx=8, pady=(8, 4))
        chart_frame.rowconfigure(0, weight=1)
        chart_frame.columnconfigure(0, weight=1)

        self._fig = Figure(figsize=(8, 3), dpi=96)
        self._ax  = self._fig.add_subplot(111)
        self._ax.set_xlabel("Elapsed time (s)")
        self._ax.set_ylabel("VRAM allocated (GB)")
        self._fig.tight_layout()

        self._canvas = FigureCanvasTkAgg(self._fig, master=chart_frame)
        self._canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        table_frame = ttk.LabelFrame(parent, text="Summary", padding=4)
        table_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=(4, 4))
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)

        cols = ("Run label", "Total time (s)", "Peak VRAM (GB)", "Avg time/chunk (s)")
        self._table = ttk.Treeview(table_frame, columns=cols, show="headings", height=5)
        for col in cols:
            self._table.heading(col, text=col)
            self._table.column(col, anchor="center", width=160)
        self._table.grid(row=0, column=0, sticky="nsew")

        sb = ttk.Scrollbar(table_frame, orient="vertical",
                           command=self._table.yview)
        self._table.configure(yscrollcommand=sb.set)
        sb.grid(row=0, column=1, sticky="ns")

        ttk.Button(parent, text="Export CSV…",
                   command=self._export_csv).grid(
            row=2, column=0, sticky="e", padx=8, pady=4)

    # ── Tab 3: Compare (stub) ─────────────────────────────────────────────────

    def _build_cmp_tab(self, parent):
        ttk.Label(
            parent,
            text="Side-by-side video comparison — Phase 1.5",
            font=("TkDefaultFont", 14),
        ).pack(expand=True)

    # ── event handlers ────────────────────────────────────────────────────────

    def _browse_video(self):
        path = filedialog.askopenfilename(
            title="Select input video",
            filetypes=[("Video", "*.mp4 *.mov *.avi *.mkv *.webm *.m4v"), ("All", "*.*")],
        )
        if path:
            self._video_var.set(path)

    def _browse_mask(self):
        path = filedialog.askopenfilename(
            title="Select mask (image or video)",
            filetypes=[
                ("Image / Video",
                 "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.mp4 *.mov *.avi"),
                ("All", "*.*"),
            ],
        )
        if path:
            self._mask_var.set(path)

    def _on_tq_toggle(self):
        self._bits_spin.config(
            state=tk.NORMAL if self._tq_var.get() else tk.DISABLED)

    def _open_results(self):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        target = os.path.abspath(RESULTS_DIR)
        if sys.platform == "win32":
            os.startfile(target)
        elif sys.platform == "darwin":
            import subprocess; subprocess.Popen(["open", target])
        else:
            import subprocess; subprocess.Popen(["xdg-open", target])

    def _on_run(self):
        if self._running:
            return
        self._running = True
        self._active_samples = []
        self._run_btn.config(state=tk.DISABLED)
        self._progress_var.set(0.0)
        self._progress_lbl.set("")
        self._log_box.config(state=tk.NORMAL)
        self._log_box.delete("1.0", tk.END)
        self._log_box.config(state=tk.DISABLED)

        kwargs = dict(
            video_path      = self._video_var.get().strip(),
            mask_path       = self._mask_var.get().strip(),
            mode            = self._mode_var.get(),
            run_label       = self._label_var.get().strip() or "run",
            neighbor_length = self._nb_var.get(),
            ref_stride      = self._rs_var.get(),
            subvideo_length = self._sv_var.get(),
            fp16            = self._fp16_var.get(),
            use_tq          = self._tq_var.get(),
            tq_bits         = self._bits_var.get(),
            mask_dilation   = self._dil_var.get(),
            flow_backbone   = self._backbone_var.get(),
            fp16_waft       = self._fp16_waft_var.get(),
            num_chunks      = self._chunks_var.get(),
            scale_h         = self._scale_h_var.get(),
            scale_w         = self._scale_w_var.get(),
            resize_ratio    = self._resize_ratio_var.get(),
            res_width       = self._res_w_var.get(),
            res_height      = self._res_h_var.get(),
            save_fps        = self._save_fps_var.get(),
            save_frames     = self._save_frames_var.get(),
            raft_iters      = self._raft_iters_var.get(),
            log_fn          = self._thread_log,
            progress_fn     = self._thread_progress,
            vram_sample_fn  = self._thread_vram_sample,
            done_fn         = self._thread_done,
        )
        threading.Thread(target=run_inpainting, kwargs=kwargs, daemon=True).start()
        self._drain_log()

    # ── thread→UI bridges ─────────────────────────────────────────────────────

    def _thread_log(self, msg: str):
        with self._log_lock:
            self._log_queue.append(msg)

    def _thread_progress(self, val: float, desc: str = ""):
        self.after(0, lambda v=val, d=desc: (
            self._progress_var.set(v),
            self._progress_lbl.set(d),
        ))

    def _thread_vram_sample(self, elapsed: float, vram_gb: float):
        self._active_samples.append((elapsed, vram_gb))

    def _thread_done(self, out_path, cmp_path, stats: dict):
        self.after(0, lambda: self._on_done(stats))

    def _drain_log(self):
        with self._log_lock:
            pending = self._log_queue[:]
            self._log_queue.clear()
        if pending:
            self._log_box.config(state=tk.NORMAL)
            for line in pending:
                self._log_box.insert(tk.END, line + "\n")
            self._log_box.see(tk.END)
            self._log_box.config(state=tk.DISABLED)
        if self._running:
            self.after(100, self._drain_log)

    def _on_done(self, stats: dict):
        # Drain remaining log lines
        self._running = False
        self._run_btn.config(state=tk.NORMAL)
        with self._log_lock:
            pending = self._log_queue[:]
            self._log_queue.clear()
        if pending:
            self._log_box.config(state=tk.NORMAL)
            for line in pending:
                self._log_box.insert(tk.END, line + "\n")
            self._log_box.see(tk.END)
            self._log_box.config(state=tk.DISABLED)

        if not stats:
            return

        # Store completed run data
        entry = {**stats, "time_series": list(self._active_samples)}
        self._bench_data.append(entry)
        self._refresh_bench()

    # ── Benchmark tab refresh ─────────────────────────────────────────────────

    def _refresh_bench(self):
        self._ax.clear()
        self._ax.set_xlabel("Elapsed time (s)")
        self._ax.set_ylabel("VRAM allocated (GB)")

        for idx, entry in enumerate(self._bench_data):
            color = _LINE_COLORS[idx % len(_LINE_COLORS)]
            ts    = entry["time_series"]
            if ts:
                xs, ys = zip(*ts)
                self._ax.plot(xs, ys, color=color,
                              label=entry["run_label"], linewidth=1.5)

        if self._bench_data:
            self._ax.legend(loc="upper left", fontsize=8)

        self._fig.tight_layout()
        self._canvas.draw()

        # Refresh summary table
        for row in self._table.get_children():
            self._table.delete(row)
        for entry in self._bench_data:
            self._table.insert("", tk.END, values=(
                entry["run_label"],
                f"{entry['total_time']:.1f}",
                f"{entry['peak_vram']:.2f}",
                f"{entry['avg_chunk_time']:.1f}",
            ))

    def _export_csv(self):
        if not self._bench_data:
            return
        path = filedialog.asksaveasfilename(
            title="Export benchmark CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All", "*.*")],
            initialfile="propainter_benchmark.csv",
        )
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["run_label", "total_time_s", "peak_vram_gb",
                             "avg_chunk_time_s", "elapsed_s", "vram_gb"])
            for entry in self._bench_data:
                ts = entry["time_series"]
                if ts:
                    for elapsed, vram in ts:
                        writer.writerow([
                            entry["run_label"],
                            f"{entry['total_time']:.3f}",
                            f"{entry['peak_vram']:.3f}",
                            f"{entry['avg_chunk_time']:.3f}",
                            f"{elapsed:.3f}",
                            f"{vram:.4f}",
                        ])
                else:
                    writer.writerow([
                        entry["run_label"],
                        f"{entry['total_time']:.3f}",
                        f"{entry['peak_vram']:.3f}",
                        f"{entry['avg_chunk_time']:.3f}",
                        "", "",
                    ])

    # ── periodic VRAM refresh ─────────────────────────────────────────────────

    def _schedule_vram(self):
        self._vram_var.set(_vram_str())
        # Also emit a live sample if a run is in progress
        if self._running:
            # approximate elapsed from sample count × 0.5s
            elapsed = len(self._active_samples) * 0.5
            self._active_samples.append((elapsed, _vram_alloc_gb()))
        # Update VRAM bar
        if torch.cuda.is_available():
            try:
                allocated = torch.cuda.memory_allocated() / 1024**3
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                pct = (allocated / total) * 100
                self._vram_bar['value'] = pct
                self._vram_label.config(
                    text=f"{allocated:.1f} / {total:.1f} GB  ({pct:.0f}%)")
            except Exception:
                pass
        self._vram_after_id = self.after(500, self._schedule_vram)

    def _on_chunks_changed(self, *args):
        try:
            chunks = self._chunks_var.get()
        except (tk.TclError, ValueError):
            chunks = 0
        if chunks > 0:
            self._sv_spin.config(state=tk.DISABLED)
        else:
            self._sv_spin.config(state="normal")

    def _on_mode_changed(self, *args):
        mode = self._mode_var.get()
        self._frame_inpaint.grid_remove()
        self._frame_outpaint.grid_remove()
        self._frame_expand.grid_remove()
        if mode == "video_inpainting":
            self._frame_inpaint.grid()
        elif mode == "video_outpainting":
            self._frame_outpaint.grid()
        else:
            self._frame_expand.grid()

    def _on_close(self):
        if self._vram_after_id is not None:
            self.after_cancel(self._vram_after_id)
        self.destroy()


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    app = App()
    app.mainloop()
