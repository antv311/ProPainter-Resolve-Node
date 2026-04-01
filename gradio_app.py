"""
ProPainter — Phase 1 Gradio Test Interface

Quality gate before the DaVinci Resolve OpenFX node.
Wraps the full inference pipeline with live logging, VRAM tracking,
TurboQuant toggle, and side-by-side comparison output.
"""

import os
import time
import threading
import tempfile
import traceback
import subprocess
import shutil
from pathlib import Path
from scipy.ndimage import binary_dilation

import cv2
import numpy as np
import imageio
from PIL import Image
import torch
import gradio as gr

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
RESULTS_DIR  = "results_gradio"

# ── model registry ─────────────────────────────────────────────────────────────
_model_lock = threading.Lock()
_models: dict = {}   # populated lazily; keys: "flow", "flow_complete", "inpaint"
_device: torch.device | None = None


def _get_device() -> torch.device:
    global _device
    if _device is None:
        _device = get_device()
    return _device


def _vram_str() -> str:
    if not torch.cuda.is_available():
        return "CPU mode (no CUDA)"
    alloc = torch.cuda.memory_allocated() / 1e9
    peak  = torch.cuda.max_memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    return f"Alloc {alloc:.2f} GB  |  Peak {peak:.2f} GB  |  Total {total:.1f} GB"


def load_models(log_fn=None):
    """
    Load all three models into the global registry (idempotent — skips already-loaded
    models).  Returns a list of (name, error_string) tuples for any failures.
    """
    global _models
    device = _get_device()

    def _log(msg):
        if log_fn:
            log_fn(msg)

    errors = []

    with _model_lock:
        # ── WAFT optical flow ─────────────────────────────────────────────────
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
                _log("    (inference will continue with zero flow; quality degraded)")
                errors.append(("flow", str(exc)))

        # ── RecurrentFlowCompleteNet ──────────────────────────────────────────
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

        # ── ProPainter InpaintGenerator ───────────────────────────────────────
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
    """Toggle TurboQuant on every SparseWindowAttention in the loaded inpaint model."""
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


def _first_frame_numpy(video_path: str) -> np.ndarray | None:
    """Extract first frame as RGB numpy array."""
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _mask_from_editor(editor_val, size: tuple) -> Image.Image | None:
    """
    Extract binary mask (mode L) from a gr.ImageEditor output dict.
    Painted regions are identified by the alpha channel of the top drawing layer.
    """
    if editor_val is None:
        return None
    layers = editor_val.get("layers") or []
    if not layers or layers[0] is None:
        return None
    layer = layers[0]
    if isinstance(layer, np.ndarray):
        alpha = layer[:, :, 3] if (layer.ndim == 3 and layer.shape[2] == 4) else (
            (layer[..., :3].mean(axis=2) > 200).astype(np.uint8) * 255
        )
    else:  # PIL
        layer = np.array(Image.fromarray(np.array(layer)).convert("RGBA"))
        alpha = layer[:, :, 3]
    mask = Image.fromarray((alpha > 127).astype(np.uint8) * 255, "L")
    return mask.resize(size, Image.NEAREST)


def _mask_from_upload(path: str | None, size: tuple) -> Image.Image | None:
    if path is None:
        return None
    return Image.open(path).convert("L").resize(size, Image.NEAREST)


def _masks_from_video(path: str, video_length: int, size: tuple,
                      flow_dilates: int = 4, mask_dilates: int = 4):
    """
    Read a per-frame mask video (MOV/MP4/etc.) via cv2.
    Returns (flow_masks, masks_dilated) as lists of PIL.Image mode 'L' of
    length video_length — same format as inference_propainter.read_mask().
    White pixels (>127) are the region to inpaint.
    If the mask video is shorter than video_length, the last frame is tiled.
    """
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    raw = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = frame[:, :, 0]  # first channel; for white/black masks any channel works
        resized = cv2.resize(gray, size, interpolation=cv2.INTER_NEAREST)
        raw.append((resized > 127).astype(np.uint8))
    cap.release()

    if not raw:
        return None, None

    # Pad to video_length if mask video is shorter, truncate if longer
    while len(raw) < video_length:
        raw.append(raw[-1].copy())
    raw = raw[:video_length]

    batch = np.stack(raw, axis=0)  # [N, H, W] uint8 0/1

    def _dilate(arr, iterations):
        if iterations <= 0:
            return arr
        return binary_dilation(arr, iterations=iterations).astype(np.uint8)

    flow_batch = _dilate(batch, flow_dilates)
    mask_batch = _dilate(batch, mask_dilates)

    flow_masks    = [Image.fromarray(f * 255, 'L') for f in flow_batch]
    masks_dilated = [Image.fromarray(m * 255, 'L') for m in mask_batch]
    return flow_masks, masks_dilated


def _write_comparison(left_frames, right_frames, fps: float, out_path: str):
    """Write side-by-side (masked | inpainted) MP4."""
    with imageio.get_writer(out_path, fps=fps, quality=7, macro_block_size=1) as w:
        for l_f, r_f in zip(left_frames, right_frames):
            l = np.array(l_f) if not isinstance(l_f, np.ndarray) else l_f
            r = np.array(r_f) if not isinstance(r_f, np.ndarray) else r_f
            w.append_data(np.concatenate([l, r], axis=1))


# ── inference generator ───────────────────────────────────────────────────────

def run_inpainting(
    video_path,
    mask_mode,
    mask_upload,
    mask_editor,
    mask_video,
    neighbor_length,
    ref_stride,
    subvideo_length,
    fp16,
    use_tq,
    tq_bits,
    mask_dilation,
    progress=gr.Progress(track_tqdm=False),
):
    """
    Generator — yields (log_text, vram_text, out_video_path, comparison_path)
    after each significant step so Gradio streams updates to the UI.
    """
    log_lines = []

    def _log(msg):
        log_lines.append(msg)

    def _state():
        return "\n".join(log_lines), _vram_str(), None, None

    def _final(out, cmp):
        return "\n".join(log_lines), _vram_str(), out, cmp

    # ── input validation ──────────────────────────────────────────────────────
    yield *_state(),

    if video_path is None:
        _log("❌  No video provided.")
        yield *_state(),
        return

    if mask_mode == "upload" and mask_upload is None:
        _log("❌  Upload mode selected but no mask image provided.")
        yield *_state(),
        return

    if mask_mode == "paint":
        layers = (mask_editor or {}).get("layers") or []
        if not layers or layers[0] is None:
            _log("❌  Paint mode selected but nothing painted. Draw the region to inpaint.")
            yield *_state(),
            return

    if mask_mode == "video" and mask_video is None:
        _log("❌  Video mask mode selected but no mask video provided.")
        yield *_state(),
        return

    # ── load models ───────────────────────────────────────────────────────────
    _log("── Loading models ─────────────────────────────────────")
    yield *_state(),
    progress(0.0, desc="Loading models…")

    errs = load_models(log_fn=_log)
    yield *_state(),

    if "flow_complete" not in _models or "inpaint" not in _models:
        _log("❌  Critical models failed to load — cannot continue.")
        yield *_state(),
        return

    # ── TurboQuant ────────────────────────────────────────────────────────────
    _patch_turboquant(bool(use_tq), int(tq_bits))
    _log(f"TurboQuant: {'ON  (bits={})'.format(int(tq_bits)) if use_tq else 'OFF'}")
    yield *_state(),

    device    = _get_device()
    use_half  = bool(fp16) and device.type == "cuda"

    # ── read video ────────────────────────────────────────────────────────────
    _log("── Reading video ──────────────────────────────────────")
    yield *_state(),
    progress(0.05, desc="Reading video…")

    try:
        frames_pil, fps = _read_video(video_path)
    except Exception:
        _log(f"❌  Failed to read video:\n{traceback.format_exc()}")
        yield *_state(),
        return

    if not frames_pil:
        _log("❌  Video contains no readable frames.")
        yield *_state(),
        return

    video_name         = Path(video_path).stem
    frames_pil, size, out_size = resize_frames(frames_pil)
    w, h               = size
    video_length       = len(frames_pil)
    frames_inp         = [np.array(f).astype(np.uint8) for f in frames_pil]

    _log(f"  {video_length} frames  |  proc {w}×{h}  |  out {out_size[0]}×{out_size[1]}  |  {fps:.2f} fps")
    yield *_state(),

    # ── build masks ───────────────────────────────────────────────────────────
    _log("── Building masks ─────────────────────────────────────")
    yield *_state(),
    progress(0.08, desc="Building masks…")

    try:
        if mask_mode == "video":
            flow_masks, masks_dilated = _masks_from_video(
                mask_video, video_length, size,
                flow_dilates=int(mask_dilation),
                mask_dilates=int(mask_dilation),
            )
            if flow_masks is None:
                _log("❌  Could not read any frames from the mask video.")
                yield *_state(),
                return
            _log(f"  Video mask: {len(flow_masks)} frames  |  dilation {mask_dilation}px")
        else:
            raw_mask = (
                _mask_from_upload(mask_upload, size)
                if mask_mode == "upload"
                else _mask_from_editor(mask_editor, size)
            )
            if raw_mask is None:
                _log("❌  Could not extract mask.")
                yield *_state(),
                return

            tmp_dir  = tempfile.mkdtemp(prefix="propainter_")
            mask_tmp = os.path.join(tmp_dir, "mask.png")
            raw_mask.save(mask_tmp)

            flow_masks, masks_dilated = read_mask(
                mask_tmp, video_length, size,
                flow_mask_dilates=int(mask_dilation),
                mask_dilates=int(mask_dilation),
            )
            _log(f"  Dilation: {mask_dilation}px  |  {len(flow_masks)} mask frames ready")
    except Exception:
        _log(f"❌  Mask processing failed:\n{traceback.format_exc()}")
        yield *_state(),
        return

    yield *_state(),

    # ── build masked-overlay frames (for comparison video) ────────────────────
    masked_for_save = []
    for frame_pil, mdil_pil in zip(frames_pil, masks_dilated):
        frame_np = np.array(frame_pil)
        msk_np   = np.expand_dims(np.array(mdil_pil), 2).repeat(3, axis=2) / 255.0
        green    = np.zeros((h, w, 3), dtype=np.float32)
        green[:, :, 1] = 255.0
        fuse     = (1.0 - 0.6) * frame_np + 0.6 * green
        masked_for_save.append((msk_np * fuse + (1.0 - msk_np) * frame_np).astype(np.uint8))

    # ── cast models to requested precision ───────────────────────────────────
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

    yield *_state(),

    # ── chunk inference loop ──────────────────────────────────────────────────
    _log("── Running inference ──────────────────────────────────")
    yield *_state(),

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

        progress(
            0.1 + 0.85 * ci / n_chunks,
            desc=f"Chunk {ci+1}/{n_chunks}  frames {chunk_start}–{chunk_end-1}",
        )
        _log(f"  Chunk {ci+1}/{n_chunks}  frames {chunk_start}–{chunk_end-1}")
        yield *_state(),

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
            # ── optical flow ─────────────────────────────────────────────────
            if frames_t.size(-1) <= 640:   scl = 12
            elif frames_t.size(-1) <= 720:  scl = 8
            elif frames_t.size(-1) <= 1280: scl = 4
            else:                           scl = 2

            if flow_model is not None:
                if frames_t.size(1) > scl:
                    fwd_list, bwd_list = [], []
                    for f in range(0, chunk_len, scl):
                        ef  = min(chunk_len, f + scl)
                        sl  = frames_t[:, f:ef] if f == 0 else frames_t[:, f - 1:ef]
                        ff, fb = flow_model(sl, iters=20)
                        fwd_list.append(ff)
                        bwd_list.append(fb)
                        torch.cuda.empty_cache()
                    gt_flows_bi = (
                        torch.cat(fwd_list, dim=1),
                        torch.cat(bwd_list, dim=1),
                    )
                else:
                    gt_flows_bi = flow_model(frames_t, iters=20)
                    torch.cuda.empty_cache()
            else:
                # WAFT weights not yet downloaded — use zero flow as fallback
                _log("    ⚠ WAFT not loaded; zero flow fallback (quality degraded)")
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

            # ── flow completion ───────────────────────────────────────────────
            pred_flows_bi, _ = flow_complete.forward_bidirect_flow(gt_flows_bi, fmasks_t)
            pred_flows_bi    = flow_complete.combine_flow(gt_flows_bi, pred_flows_bi, fmasks_t)
            torch.cuda.empty_cache()

            # ── image propagation ─────────────────────────────────────────────
            masked_f = frames_t * (1 - mdil_t)
            b, t, _, _, _ = mdil_t.size()
            prop_imgs, upd_masks = inpaint_model.img_propagation(
                masked_f, pred_flows_bi, mdil_t, "nearest"
            )
            upd_frames = frames_t * (1 - mdil_t) + prop_imgs.view(b, t, 3, h, w) * mdil_t
            upd_masks  = upd_masks.view(b, t, 1, h, w)
            torch.cuda.empty_cache()

        # ── feature propagation + transformer ────────────────────────────────
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
        vram_peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        _log(f"    ✓ {elapsed:.1f}s  |  peak VRAM {vram_peak:.2f} GB")
        yield *_state(),

        del frames_t, fmasks_t, mdil_t, gt_flows_bi, pred_flows_bi
        del masked_f, prop_imgs, upd_frames, upd_masks
        torch.cuda.empty_cache()

    # ── normalise blended frames ──────────────────────────────────────────────
    for i in range(video_length):
        if comp_frames[i] is None:
            comp_frames[i] = frames_inp[i]          # uncovered frame → original
        elif comp_weights[i] > 0:
            comp_frames[i] = (comp_frames[i] / comp_weights[i]).astype(np.uint8)

    total_time = time.perf_counter() - wall_t0
    _log(f"── Complete  {total_time:.1f}s total  [{_vram_str()}]")
    yield *_state(),

    # ── write output videos ───────────────────────────────────────────────────
    progress(0.97, desc="Writing output videos…")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    out_mp4 = os.path.join(RESULTS_DIR, f"{video_name}_inpainted.mp4")
    cmp_mp4 = os.path.join(RESULTS_DIR, f"{video_name}_comparison.mp4")

    comp_out   = [cv2.resize(f, out_size) for f in comp_frames]
    masked_out = [cv2.resize(f, out_size) for f in masked_for_save]

    _log(f"  Writing inpainted:  {len(comp_out)} frames  "
         f"shape={comp_out[0].shape}  fps={fps:.2f}")
    imageio.mimwrite(out_mp4, comp_out, fps=fps, quality=7, macro_block_size=1)
    out_mb = os.path.getsize(out_mp4) / 1e6 if os.path.exists(out_mp4) else 0.0
    _log(f"  → {out_mp4}  ({out_mb:.2f} MB, exists={os.path.exists(out_mp4)})")

    _log(f"  Writing comparison: {len(masked_out)} frames  "
         f"shape={masked_out[0].shape}  fps={fps:.2f}")
    _write_comparison(masked_out, comp_out, fps, cmp_mp4)
    cmp_mb = os.path.getsize(cmp_mp4) / 1e6 if os.path.exists(cmp_mp4) else 0.0
    _log(f"  → {cmp_mp4}  ({cmp_mb:.2f} MB, exists={os.path.exists(cmp_mp4)})")

    progress(1.0, desc="Done")

    yield *_final(out_mp4, cmp_mp4),


# ── ffmpeg transcode helper ───────────────────────────────────────────────────

_FFMPEG_FALLBACK = r'C:\tools\ffmpeg\bin\ffmpeg.exe'


def _ffprobe_codec(input_path: str, ffmpeg_exe: str) -> str | None:
    """Return the video codec name (e.g. 'h264', 'hevc') or None on failure."""
    ffprobe = os.path.join(os.path.dirname(ffmpeg_exe), 'ffprobe.exe') if os.name == 'nt' else \
              os.path.join(os.path.dirname(ffmpeg_exe), 'ffprobe')
    if not os.path.isfile(ffprobe):
        ffprobe = shutil.which('ffprobe') or ffprobe
    try:
        result = subprocess.run(
            [ffprobe, '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=codec_name',
             '-of', 'default=noprint_wrappers=1', input_path],
            capture_output=True, text=True, timeout=30,
        )
        for line in result.stdout.splitlines():
            if line.startswith('codec_name='):
                return line.split('=', 1)[1].strip()
    except Exception:
        pass
    return None


def _ffmpeg_transcode(input_path: str):
    """
    Probe input_path with ffprobe; skip transcode if already H.264.
    Otherwise transcode to H.264 / yuv420p MP4 for browser compatibility.
    Returns (output_path, success, log_lines).
    output_path is the H.264 path on success, or input_path on failure.
    """
    ffmpeg = shutil.which('ffmpeg') or (_FFMPEG_FALLBACK if os.path.isfile(_FFMPEG_FALLBACK) else None)

    lines = []
    if ffmpeg is None:
        lines.append("⚠ ffmpeg not found — checked PATH and C:\\tools\\ffmpeg\\bin\\ffmpeg.exe")
        return input_path, False, lines

    lines.append(f"ffmpeg: {ffmpeg}")

    # ── ffprobe: skip transcode if already H.264 ─────────────────────────────
    codec = _ffprobe_codec(input_path, ffmpeg)
    lines.append(f"ffprobe: codec_name={codec}")
    if codec == 'h264':
        lines.append("Already H.264 — skipping transcode")
        return input_path, True, lines

    # ── transcode ─────────────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, Path(input_path).stem + "_h264.mp4")
    cmd = [
        ffmpeg, '-y', '-i', input_path,
        '-c:v', 'libx264', '-crf', '18', '-preset', 'fast',
        '-pix_fmt', 'yuv420p', '-c:a', 'copy',
        output_path,
    ]
    lines.append(f"cmd: {' '.join(cmd)}")
    try:
        proc = subprocess.Popen(
            cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL,
            text=True, encoding='utf-8', errors='replace',
        )
        for stderr_line in proc.stderr:
            lines.append(stderr_line.rstrip())
        proc.wait(timeout=300)
        lines.append(f"ffmpeg exit: {proc.returncode}")
        if proc.returncode == 0:
            out_mb = os.path.getsize(output_path) / 1e6 if os.path.exists(output_path) else 0.0
            lines.append(f"Transcode OK → {output_path}  ({out_mb:.2f} MB)")
            return output_path, True, lines
        else:
            lines.append("⚠ ffmpeg failed — using original file")
            return input_path, False, lines
    except subprocess.TimeoutExpired:
        proc.kill()
        lines.append("ffmpeg timed out after 300 s")
        return input_path, False, lines
    except Exception as exc:
        lines.append(f"ffmpeg error: {exc}")
        return input_path, False, lines


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="ProPainter — Phase 1 Test Interface",
        theme=gr.themes.Base(),
    ) as demo:

        gr.Markdown(
            "# ProPainter — Phase 1 Test Interface\n"
            "Quality gate for the DaVinci Resolve inference pipeline.  "
            "Upload a video, mark the region to inpaint, tune settings, run."
        )

        with gr.Row(equal_height=False):

            # ── LEFT COLUMN: inputs ──────────────────────────────────────────
            with gr.Column(scale=1, min_width=420):

                video_input = gr.Video(
                    label="Input video",
                    sources=["upload"],
                    format="mp4",
                    height=280,
                )

                with gr.Row():
                    codec_info_md = gr.Markdown(visible=False, scale=3)
                    convert_btn   = gr.Button(
                        "Convert to H.264 for preview",
                        visible=False,
                        scale=1,
                        variant="secondary",
                    )

                gr.Markdown("### Mask")
                mask_mode = gr.Radio(
                    choices=["upload", "paint", "video"],
                    value="upload",
                    label="Mask input mode",
                )

                mask_upload = gr.Image(
                    label="Mask image  (white = region to inpaint, black = keep)",
                    type="filepath",
                    image_mode="L",
                    visible=True,
                )

                mask_editor = gr.ImageEditor(
                    label="Paint over the region to inpaint  (white brush)",
                    brush=gr.Brush(colors=["#ffffff"], color_mode="fixed"),
                    eraser=gr.Eraser(default_size=20),
                    type="numpy",
                    height=300,
                    visible=False,
                )

                mask_video = gr.Video(
                    label="Mask video  (white = inpaint, black = keep — MOV/MP4)",
                    sources=["upload"],
                    visible=False,
                )

                with gr.Accordion("Settings", open=True):
                    with gr.Row():
                        neighbor_length = gr.Slider(
                            4, 30, value=10, step=2,
                            label="Neighbor length",
                            info="Local temporal window size",
                        )
                        ref_stride = gr.Slider(
                            2, 20, value=10, step=1,
                            label="Reference stride",
                            info="Global reference frame spacing",
                        )
                    with gr.Row():
                        subvideo_length = gr.Slider(
                            20, 200, value=80, step=10,
                            label="Subvideo length",
                            info="Chunk size (frames) — reduce for OOM",
                        )
                        mask_dilation = gr.Slider(
                            0, 10, value=4, step=1,
                            label="Mask dilation (px)",
                        )
                    with gr.Row():
                        fp16_toggle = gr.Checkbox(value=True,  label="FP16  (half precision)")
                        tq_toggle   = gr.Checkbox(value=False, label="TurboQuant KV compression")
                    tq_bits = gr.Slider(
                        2, 4, value=3, step=1,
                        label="TurboQuant bits  (2 / 3 / 4)",
                        visible=False,
                    )

                run_btn = gr.Button("▶  Run Inpainting", variant="primary", size="lg")

            # ── RIGHT COLUMN: outputs ────────────────────────────────────────
            with gr.Column(scale=1, min_width=420):

                with gr.Row():
                    vram_display = gr.Textbox(
                        value=_vram_str(),
                        label="VRAM",
                        interactive=False,
                        lines=1,
                        scale=3,
                    )
                    vram_timer = gr.Timer(value=2)

                upload_log = gr.Textbox(
                    label="Upload log",
                    lines=5,
                    autoscroll=True,
                    interactive=False,
                )

                log_box = gr.Textbox(
                    label="Log",
                    lines=10,
                    autoscroll=True,
                    interactive=False,
                )

                out_video = gr.Video(label="Inpainted output")
                cmp_video = gr.Video(label="Side-by-side comparison  (masked | inpainted)")

        # ── event wiring ──────────────────────────────────────────────────────

        # ── State: working file paths (what inference actually reads) ─────────
        working_path      = gr.State(value=None)
        mask_working_path = gr.State(value=None)

        # When video is uploaded: copy to RESULTS_DIR, probe, update state+UI
        def _on_video_upload(video_path):
            """
            Returns: (mask_editor, upload_log, codec_row, convert_btn, working_path)
            Never touches video_path after copying — avoids Gradio temp-file locks.
            """
            if video_path is None:
                return gr.update(), "", gr.update(visible=False), gr.update(visible=False), None

            os.makedirs(RESULTS_DIR, exist_ok=True)
            dest = os.path.join(RESULTS_DIR, Path(video_path).name)
            shutil.copy2(video_path, dest)

            lines = [f"Original: {video_path}", f"Copied:   {dest}"]
            try:
                lines.append(f"Size:  {os.path.getsize(dest) / 1e6:.2f} MB")
            except OSError as e:
                lines.append(f"Size:  ERROR — {e}")

            cap = cv2.VideoCapture(dest, cv2.CAP_FFMPEG)
            lines.append(f"cv2.isOpened():  {cap.isOpened()}")
            if cap.isOpened():
                lines.append(
                    f"Frames: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}  |  "
                    f"FPS: {cap.get(cv2.CAP_PROP_FPS):.2f}  |  "
                    f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}×"
                    f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
                )
            cap.release()

            # Probe codec on the copy
            ffmpeg = shutil.which('ffmpeg') or (_FFMPEG_FALLBACK if os.path.isfile(_FFMPEG_FALLBACK) else None)
            codec = _ffprobe_codec(dest, ffmpeg) if ffmpeg else None
            lines.append(f"Codec: {codec or 'unknown'}")

            if codec == 'h264':
                codec_md  = gr.update(value="✓ H.264 — no conversion needed", visible=True)
                conv_btn  = gr.update(visible=False)
            else:
                codec_md  = gr.update(value=f"⚠ Codec: **{codec or 'unknown'}** — convert for browser preview", visible=True)
                conv_btn  = gr.update(visible=True)

            editor_update = gr.update()
            frame = _first_frame_numpy(dest)
            if frame is not None:
                editor_update = gr.update(
                    value={"background": frame, "layers": [], "composite": frame}
                )

            return editor_update, "\n".join(lines), codec_md, conv_btn, dest

        video_input.change(
            fn=_on_video_upload,
            inputs=[video_input],
            outputs=[mask_editor, upload_log, codec_info_md, convert_btn, working_path],
        )

        # Convert button: transcode the working copy, update state + video preview
        def _on_convert_click(current_path):
            if current_path is None:
                return "No file loaded.", gr.update(), gr.update(visible=True), None
            h264_path, ok, tx_lines = _ffmpeg_transcode(current_path)
            log_text = "\n".join(tx_lines)
            if ok:
                return log_text, gr.update(value=h264_path), gr.update(visible=False), h264_path
            else:
                return log_text + "\n⚠ Conversion failed — using original", gr.update(), gr.update(visible=True), current_path

        convert_btn.click(
            fn=_on_convert_click,
            inputs=[working_path],
            outputs=[upload_log, video_input, convert_btn, working_path],
        )

        # When mask video is uploaded: copy to RESULTS_DIR, probe, auto-transcode
        def _on_mask_video_upload(video_path):
            if video_path is None:
                return "", gr.update(), None

            os.makedirs(RESULTS_DIR, exist_ok=True)
            dest = os.path.join(RESULTS_DIR, "mask_" + Path(video_path).name)
            shutil.copy2(video_path, dest)

            lines = [f"Mask copied: {dest}"]
            h264_path, ok, tx_lines = _ffmpeg_transcode(dest)
            lines.extend(tx_lines)
            if not ok:
                lines.append("⚠ Using original file")

            return "\n".join(lines), gr.update(value=h264_path), h264_path

        mask_video.change(
            fn=_on_mask_video_upload,
            inputs=[mask_video],
            outputs=[upload_log, mask_video, mask_working_path],
        )

        # Switch between upload / paint / video mask inputs
        def _on_mask_mode(mode):
            return (
                gr.update(visible=mode == "upload"),
                gr.update(visible=mode == "paint"),
                gr.update(visible=mode == "video"),
            )

        mask_mode.change(
            fn=_on_mask_mode,
            inputs=[mask_mode],
            outputs=[mask_upload, mask_editor, mask_video],
        )

        # Show / hide TurboQuant bits slider
        tq_toggle.change(
            fn=lambda v: gr.update(visible=v),
            inputs=[tq_toggle],
            outputs=[tq_bits],
        )

        # Periodic VRAM refresh
        vram_timer.tick(fn=_vram_str, outputs=[vram_display])

        # Main inference run
        run_btn.click(
            fn=run_inpainting,
            inputs=[
                working_path, mask_mode, mask_upload, mask_editor, mask_working_path,
                neighbor_length, ref_stride, subvideo_length,
                fp16_toggle, tq_toggle, tq_bits, mask_dilation,
            ],
            outputs=[log_box, vram_display, out_video, cmp_video],
        )

    return demo


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    app = build_ui()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
    )
