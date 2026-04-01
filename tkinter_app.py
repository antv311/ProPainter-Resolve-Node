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
from inference_propainter import resize_frames, read_mask, get_ref_index
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


def load_models(log_fn=None):
    global _models
    device = _get_device()

    def _log(msg):
        if log_fn:
            log_fn(msg)

    errors = []

    with _model_lock:
        if "flow" not in _models:
            try:
                _log("  Loading WAFT flow model…")
                ckpt = load_file_from_url(
                    url=os.path.join(PRETRAIN_URL, "waft-downstream.pth"),
                    model_dir=WEIGHTS_DIR, progress=False, file_name=None,
                )
                _models["flow"] = WAFT_bi(ckpt, device)
                _log(f"  ✓ WAFT  [{_vram_str()}]")
            except Exception as exc:
                _log(f"  ✗ WAFT failed — {exc}")
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
    run_label: str,
    neighbor_length: int,
    ref_stride: int,
    subvideo_length: int,
    fp16: bool,
    use_tq: bool,
    tq_bits: int,
    mask_dilation: int,
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
    def _log(msg):
        log_fn(msg)

    def _prog(val, desc=""):
        progress_fn(val, desc)

    if not video_path or not os.path.isfile(video_path):
        _log("❌  Video file not found.")
        done_fn(None, None, {})
        return

    if not mask_path or not os.path.isfile(mask_path):
        _log("❌  Mask file not found.")
        done_fn(None, None, {})
        return

    run_label      = run_label.strip() or "run"
    mask_is_video  = Path(mask_path).suffix.lower() in _VIDEO_EXTS
    peak_vram      = 0.0
    chunk_times: list[float] = []

    # ── load models ───────────────────────────────────────────────────────────
    _log("── Loading models ─────────────────────────────────────")
    _prog(0.0, "Loading models…")
    load_models(log_fn=_log)

    if "flow_complete" not in _models or "inpaint" not in _models:
        _log("❌  Critical models failed — cannot continue.")
        done_fn(None, None, {})
        return

    _patch_turboquant(bool(use_tq), int(tq_bits))
    _log(f"TurboQuant: {'ON  bits={}'.format(int(tq_bits)) if use_tq else 'OFF'}")

    device   = _get_device()
    use_half = bool(fp16) and device.type == "cuda"

    # ── read video ────────────────────────────────────────────────────────────
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

    video_name              = Path(video_path).stem
    frames_pil, size, out_size = resize_frames(frames_pil)
    w, h                    = size
    video_length            = len(frames_pil)
    frames_inp              = [np.array(f).astype(np.uint8) for f in frames_pil]
    _log(f"  {video_length} frames | proc {w}×{h} | out {out_size[0]}×{out_size[1]} | {fps:.2f} fps")

    # ── build masks ───────────────────────────────────────────────────────────
    _log("── Building masks ─────────────────────────────────────")
    _prog(0.08, "Building masks…")
    try:
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

    # ── masked overlay frames for comparison video ────────────────────────────
    masked_for_save = []
    for fp2, mdil_pil in zip(frames_pil, masks_dilated):
        frame_np = np.array(fp2)
        msk_np   = np.expand_dims(np.array(mdil_pil), 2).repeat(3, axis=2) / 255.0
        green    = np.zeros((h, w, 3), dtype=np.float32)
        green[:, :, 1] = 255.0
        fuse     = 0.4 * frame_np + 0.6 * green
        masked_for_save.append((msk_np * fuse + (1.0 - msk_np) * frame_np).astype(np.uint8))

    # ── precision cast ────────────────────────────────────────────────────────
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

    # ── chunk inference loop ──────────────────────────────────────────────────
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

        frames_t = to_tensors()(c_frames).unsqueeze(0) * 2 - 1
        fmasks_t = to_tensors()(c_fmasks).unsqueeze(0)
        mdil_t   = to_tensors()(c_mdil).unsqueeze(0)
        frames_t = frames_t.to(device)
        fmasks_t = fmasks_t.to(device)
        mdil_t   = mdil_t.to(device)

        chunk_t0 = time.perf_counter()

        with torch.no_grad():
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
                        ff, fb = flow_model(sl, iters=20)
                        fwd_list.append(ff)
                        bwd_list.append(fb)
                        torch.cuda.empty_cache()
                    gt_flows_bi = (torch.cat(fwd_list, dim=1), torch.cat(bwd_list, dim=1))
                else:
                    gt_flows_bi = flow_model(frames_t, iters=20)
                    torch.cuda.empty_cache()
            else:
                _log("    ⚠ WAFT not loaded — zero flow fallback")
                B, T, C, H, W = frames_t.shape
                gt_flows_bi = (
                    torch.zeros(B, T - 1, 2, H, W, device=device),
                    torch.zeros(B, T - 1, 2, H, W, device=device),
                )

            if use_half:
                frames_t    = frames_t.half()
                fmasks_t    = fmasks_t.half()
                mdil_t      = mdil_t.half()
                gt_flows_bi = (gt_flows_bi[0].half(), gt_flows_bi[1].half())

            pred_flows_bi, _ = flow_complete.forward_bidirect_flow(gt_flows_bi, fmasks_t)
            pred_flows_bi    = flow_complete.combine_flow(gt_flows_bi, pred_flows_bi, fmasks_t)
            torch.cuda.empty_cache()

            masked_f = frames_t * (1 - mdil_t)
            b, t, _, _, _ = mdil_t.size()
            prop_imgs, upd_masks = inpaint_model.img_propagation(
                masked_f, pred_flows_bi, mdil_t, "nearest"
            )
            upd_frames = frames_t * (1 - mdil_t) + prop_imgs.view(b, t, 3, h, w) * mdil_t
            upd_masks  = upd_masks.view(b, t, 1, h, w)
            torch.cuda.empty_cache()

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

            with torch.no_grad():
                l_t  = len(nb_ids)
                pred = inpaint_model(sel_imgs, sel_flows, sel_masks, sel_upd_mask, l_t)
                pred = pred.view(-1, 3, h, w)
                pred = (pred + 1) / 2
                pred_np   = pred.cpu().permute(0, 2, 3, 1).numpy() * 255
                bin_masks = mdil_t[0, nb_ids].cpu().permute(0, 2, 3, 1).numpy().astype(np.uint8)

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

            torch.cuda.empty_cache()

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

    # ── normalise blended frames ──────────────────────────────────────────────
    for i in range(video_length):
        if comp_frames[i] is None:
            comp_frames[i] = frames_inp[i]
        elif comp_weights[i] > 0:
            comp_frames[i] = (comp_frames[i] / comp_weights[i]).astype(np.uint8)

    total_time = time.perf_counter() - wall_t0
    _log(f"── Complete  {total_time:.1f}s total  [{_vram_str()}]")

    # ── write output videos ───────────────────────────────────────────────────
    _prog(0.97, "Writing output videos…")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    safe_label = run_label.replace(" ", "_")
    out_mp4    = os.path.join(RESULTS_DIR, f"{video_name}_{safe_label}_inpainted.mp4")
    cmp_mp4    = os.path.join(RESULTS_DIR, f"{video_name}_{safe_label}_comparison.mp4")

    comp_out   = [cv2.resize(f, out_size) for f in comp_frames]
    masked_out = [cv2.resize(f, out_size) for f in masked_for_save]

    _log(f"  Writing {out_mp4}")
    imageio.mimwrite(out_mp4, comp_out, fps=fps, quality=7, macro_block_size=1)
    _log(f"  Writing {cmp_mp4}")
    _write_comparison(masked_out, comp_out, fps, cmp_mp4)

    out_mb = os.path.getsize(out_mp4) / 1e6 if os.path.exists(out_mp4) else 0.0
    cmp_mb = os.path.getsize(cmp_mp4) / 1e6 if os.path.exists(cmp_mp4) else 0.0
    _log(f"  ✓ inpainted  {out_mb:.2f} MB")
    _log(f"  ✓ comparison {cmp_mb:.2f} MB")

    _prog(1.0, "Done")

    stats = {
        "run_label":      run_label,
        "total_time":     total_time,
        "peak_vram":      peak_vram,
        "chunk_times":    chunk_times,
        "avg_chunk_time": sum(chunk_times) / len(chunk_times) if chunk_times else 0.0,
    }
    done_fn(out_mp4, cmp_mp4, stats)


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

        self._build_ui()
        self._schedule_vram()

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

        def _file_row(label, var_attr, browse_cmd):
            nonlocal row
            ttk.Label(p, text=label).grid(
                row=row, column=0, columnspan=3, sticky="w", pady=(6, 0))
            row += 1
            var = tk.StringVar()
            setattr(self, var_attr, var)
            ttk.Entry(p, textvariable=var).grid(
                row=row, column=0, columnspan=2, sticky="ew", pady=2)
            ttk.Button(p, text="Browse…", command=browse_cmd, width=8).grid(
                row=row, column=2, padx=(4, 0), pady=2)
            row += 1

        _file_row("Video path:", "_video_var", self._browse_video)
        _file_row("Mask path  (image or video):", "_mask_var", self._browse_mask)

        ttk.Separator(p, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=8)
        row += 1

        ttk.Label(p, text="Run label:").grid(row=row, column=0, sticky="w")
        self._label_var = tk.StringVar(value="baseline")
        ttk.Entry(p, textvariable=self._label_var).grid(
            row=row, column=1, columnspan=2, sticky="ew", padx=(4, 0))
        row += 1

        ttk.Separator(p, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=8)
        row += 1

        def _spin(label, var_attr, default, lo, hi):
            nonlocal row
            var = tk.IntVar(value=default)
            setattr(self, var_attr, var)
            ttk.Label(p, text=label).grid(row=row, column=0, sticky="w", pady=2)
            ttk.Spinbox(p, from_=lo, to=hi, textvariable=var, width=6).grid(
                row=row, column=1, sticky="w", padx=(4, 0), pady=2)
            row += 1

        _spin("Neighbor length:",    "_nb_var",  10,  4,  30)
        _spin("Reference stride:",   "_rs_var",  10,  2,  20)
        _spin("Subvideo length:",    "_sv_var",  80, 20, 200)
        _spin("Mask dilation (px):", "_dil_var",  4,  0,  20)

        ttk.Separator(p, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=8)
        row += 1

        self._fp16_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(p, text="FP16  (half precision)",
                        variable=self._fp16_var).grid(
            row=row, column=0, columnspan=3, sticky="w")
        row += 1

        self._tq_var   = tk.BooleanVar(value=False)
        self._bits_var = tk.IntVar(value=3)
        tq_row = ttk.Frame(p)
        tq_row.grid(row=row, column=0, columnspan=3, sticky="w", pady=2)
        ttk.Checkbutton(tq_row, text="TurboQuant",
                        variable=self._tq_var,
                        command=self._on_tq_toggle).pack(side=tk.LEFT)
        ttk.Label(tq_row, text="  bits:").pack(side=tk.LEFT)
        self._bits_spin = ttk.Spinbox(tq_row, from_=2, to=4,
                                      textvariable=self._bits_var, width=4)
        self._bits_spin.pack(side=tk.LEFT, padx=(2, 0))
        self._on_tq_toggle()
        row += 1

        ttk.Separator(p, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=8)
        row += 1

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
            run_label       = self._label_var.get().strip() or "run",
            neighbor_length = self._nb_var.get(),
            ref_stride      = self._rs_var.get(),
            subvideo_length = self._sv_var.get(),
            fp16            = self._fp16_var.get(),
            use_tq          = self._tq_var.get(),
            tq_bits         = self._bits_var.get(),
            mask_dilation   = self._dil_var.get(),
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
        self.after(500, self._schedule_vram)


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    app = App()
    app.mainloop()
