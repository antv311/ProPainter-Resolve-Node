# CLAUDE.md — ProPainter-Resolve-Node
Last updated: 2026-04-01

---

## Who is Tony

VFX artist running a fully local pipeline. Two client service areas:
- **Memorial video restoration** — converting and restoring low-quality vertical footage for clients who have lost loved ones
- **Aspect ratio conversion** — 9:16 → 16:9 for content creators moving from short-form to long-form platforms

**Non-negotiable constraint:** No cloud processing. DoD-grade drive wipes after every job. Everything stays local.

**Hardware:** RTX 3080 (8GB VRAM), Windows 11, fully local
**NLE:** DaVinci Resolve Studio (with Fusion and Fairlight)

---

## Project Goal

Build a fully local, automated VFX pipeline inside DaVinci Resolve Studio replacing a sophisticated but labor-intensive manual workflow. The first inference tool is ProPainter for video inpainting and object removal.

---

## Phase Structure

### Phase 1 — Get the AI Guts Working ✅ COMPLETE
Get the inference stack fully modernized and running on Python 3.14 + CUDA 13.1 + WAFT + TurboQuant. Validated through a Tkinter benchmarking harness. The Resolve node comes later.

Deliverables:
- Python 3.14 venv with source-built PyTorch (CUDA 13.1) ✅ Wheel built & cached
- torchvision ✅ ELIMINATED — deform_conv2d replaced with tvdcn 1.1.0; read_video replaced with cv2
- opencv-python source build ✅ Wheel built & cached
- xformers source build ✅ Wheel built & cached
- WAFT replacing RAFT as the optical flow backbone ✅ Complete (waft-downstream.pth downloaded; backbone combobox in tkinter_app.py)
- TurboQuant integrated into ProPainter's spatiotemporal attention layers ✅ Complete (cosine sim 0.9675)
- tvdcn wired into recurrent_flow_completion.py ✅ Complete
- Gradio UI ✅ Built, then replaced — see tkinter_app.py
- tkinter_app.py ✅ Three-tab benchmarking harness (Run / Benchmark / Compare stub)
- All dead training/eval code removed from the repo ⬜ Pending (datasets/, train.py, configs/, scripts/, web-demos/ staged for deletion)

### Phase 1.1 — Per-Clip Mask Trimming for Partial Frame Exits (Separate Node)

**Problem:** Subject exits frame mid-clip (e.g. frames 3–6). The mask needs to be active only on the frames where the subject is absent or partially absent — not the full clip. ProPainter handles the reconstruction natively via temporal propagation from surrounding frames.

**Goal:** A separate Resolve node (or pre-processing step) that:
- Takes a frame range input (e.g. "frames 3–6")
- Activates the inpaint mask only on those frames
- Passes the trimmed per-frame mask sequence to ProPainter

**Why separate node:** The main ProPainter node operates on a single static or animated mask. Partial-exit clips need a mask that turns on and off at specific timecodes — cleaner as a dedicated node than complicating the main node's mask logic.

**Status:** ⬜ Not started — revisit after Phase 2 node architecture is established.

### Phase 1.5 — Side-by-Side Video Comparison (Tab 3 stub)

**Goal:** Populate the Compare tab in `tkinter_app.py` with a true side-by-side video player for A/B review of results from different run labels.

**Status:** ⬜ Stub only ("Side-by-side video comparison — Phase 1.5" label). Full implementation deferred until Phase 1 inference quality is validated.

### Phase 2 — DaVinci Resolve Node
OpenFX C++ plugin talking to persistent Python inference servers over Unix sockets. Models stay resident in memory between frames — per-frame reload is a non-starter for performance. This phase starts only after Phase 1 is stable and validated.

---

## Repo

**Primary fork:** `antv311/ProPainter-Resolve-Node`
(forked from `sczhou/ProPainter`)

**Flow backbone:** `princeton-vl/WAFT` (`waftv2` branch)

---

## Directory Structure

```
C:\Users\tony\
├── venvs\
│   ├── venvbp\        ← reference venv where all source builds happen
│   ├── propainter\    ← per-project venv (install from wheel cache)
│   └── scr\           ← source repos for builds
│       ├── pytorch\
│       ├── torchvision\
│       ├── opencv-python\
│       └── xformers\
└── venvs\wheels\      ← local wheel cache — reuse for every new node
    ├── torch-2.12.0a0+gitfafc7d6-cp314-cp314-win_amd64.whl          ✅
    ├── opencv_python-4.13.0.92-cp314-cp314-win_amd64.whl             ✅
    ├── xformers-0.0.35+6e9337ce.d20260329-py39-none-win_amd64.whl    ✅
    └── requirements-venvbp-frozen.txt                                 ✅
```

**Note:** Wheel directory was changed from `C:\Users\tony\wheels\` to `C:\Users\tony\venvs\wheels\` during setup.

**Workflow for new nodes:** fresh venv + `pip install` from `wheels\` — no recompilation needed.

---

## Build Stack

| Component | Target | Status |
|-----------|--------|--------|
| Python | 3.14.2 | ✅ Installed |
| Git | 2.51.2 | ✅ Installed |
| CUDA Toolkit | 13.1 | ✅ Installed (`v13.1.80`) |
| Visual Studio 2022 Build Tools | 17.14 (Fall 2024 LTSC) | ✅ Installed |
| Visual Studio 2026 Build Tools | v18.4.2 (also present) | ✅ Installed (not used for CUDA builds) |
| CMake | 4.3.1 | ✅ Installed (`C:\tools\cmake\cmake-4.3.1\bin`) |
| Ninja | 1.13.2 | ✅ Installed (`C:\tools\ninja`) |
| PyTorch | 2.12.0a0+gitfafc7d6, cp314, win_amd64 | ✅ Wheel built & cached |
| torchvision | ELIMINATED | ✅ deform_conv2d → tvdcn 1.1.0; read_video → cv2 |
| opencv-python | 4.13.0.92, cp314, win_amd64 | ✅ Wheel built & cached |
| xformers | 0.0.35+6e9337ce, cp314 (tagged py39), win_amd64 | ✅ Wheel built & cached |

**Install order matters:** CUDA → VS Build Tools → CMake → Ninja → source builds

---

## venvbp Full Environment (pip freeze as of 2026-03-29)

Key packages in venvbp:

| Package | Version |
|---------|---------|
| torch | 2.12.0a0+gitfafc7d6 (local wheel) |
| opencv-python | 4.13.0.92 (local wheel) |
| xformers | 0.0.35+6e9337ce.d20260329 (local wheel) |
| triton-windows | 3.6.0.post26 |
| numpy | 2.4.4 |
| scipy | 1.17.1 |
| pillow | 12.1.1 |
| safetensors | 0.7.0 |
| transformers | 5.4.0 |
| huggingface-hub | 1.8.0 |
| imageio | 2.37.3 |
| imageio-ffmpeg | 0.6.0 |
| matplotlib | (for tkinter benchmark chart) |

Full frozen requirements saved at: `C:\Users\tony\venvs\wheels\requirements-venvbp-frozen.txt`

---

## Installing from Wheel Cache (New Venv Procedure)

```powershell
# Create new venv
python -m venv C:\Users\tony\venvs\propainter
.\propainter\Scripts\activate

# Install source-built wheels (no-deps to prevent PyPI torch clobbering our wheel)
pip install --no-deps C:\Users\tony\venvs\wheels\torch-2.12.0a0+gitfafc7d6-cp314-cp314-win_amd64.whl
pip install --no-deps C:\Users\tony\venvs\wheels\opencv_python-4.13.0.92-cp314-cp314-win_amd64.whl
pip install --no-deps C:\Users\tony\venvs\wheels\xformers-0.0.35+6e9337ce.d20260329-py39-none-win_amd64.whl

# Then install remaining deps normally
pip install triton-windows scipy pillow safetensors transformers huggingface-hub imageio imageio-ffmpeg matplotlib
```

**CRITICAL:** Always use `--no-deps` when installing our custom wheels. Never use `--force-reinstall` on xformers — it will pull torch 2.11.0+cpu from PyPI and nuke the custom wheel.

---

## torchvision — SKIPPED

ProPainter uses torchvision for exactly one call: `torchvision.io.read_video` in
`inference_propainter.py`. This is already on the modernization list to replace with
`cv2.VideoCapture`. Since that's the only usage, we skip the torchvision source build
entirely and handle it during the ProPainter modernization pass.

The torchvision CUDA build hit a persistent `bool char` reserved keyword conflict in
`c10/cuda/CUDACachingAllocator.h` — a real upstream bug where nvcc 13.1 rejects `char`
as a parameter name. The fix (rename to `bool small`) is a valid upstream PR target for
the pytorch repo.

---

## Build Environment — Critical Notes

### CUDA 13.1 + VS 2022 compatibility
CUDA 13.1 only supports MSVC up to VS 2022 (v17.x). VS 2026 (v18) is installed but
**must not be used for CUDA compilation** — it causes `chrono` header parse errors.

**Required env setup before any CUDA source build (run in PowerShell):**

```powershell
# Import full VS 2022 environment (compiler + headers + libs)
cmd /c "`"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat`" && set" | ForEach-Object {
    if ($_ -match "^([^=]+)=(.*)$") {
        [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2])
    }
}

# Then set these
$env:CUDAHOSTCXX="C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe"
$env:CUDA_HOME="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"
$env:CUDA_PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"
$env:TORCH_CUDA_ARCH_LIST="8.6"
$env:MAX_JOBS="4"
$env:DISTUTILS_USE_SDK=1
```

**Why vcvars64 matters:** Setting only `CUDAHOSTCXX` points nvcc at the right compiler binary
but leaves VS 2026 headers on the include path. The `vcvars64.bat` import replaces the entire
VS environment (includes, libs, PATH) with VS 2022 versions. Both steps are required.

### CUDA DLL locations
CUDA 13.1 puts DLLs in a non-standard location. Both paths must be on PATH permanently:

```powershell
# Add permanently to system PATH (run once, requires admin):
[System.Environment]::SetEnvironmentVariable("PATH",
  "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64;" +
  "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\extras\CUPTI\lib64;" +
  [System.Environment]::GetEnvironmentVariable("PATH", "Machine"), "Machine")
```

Key DLLs live in `bin\x64` (not `bin`) and `extras\CUPTI\lib64`. Without these,
`import torch` fails with `WinError 126` on `aoti_custom_ops.dll`.

### Windows DLL unblocking
DLLs downloaded from the internet may be blocked by Windows (Mark of the Web).
Unblock all torch DLLs after installing from a wheel:

```powershell
Get-ChildItem "C:\Users\tony\venvs\venvbp\Lib\site-packages\torch" -Recurse -Filter "*.dll" | Unblock-File
```

### GPU target
RTX 3080 = Ampere = `sm_86`. Always set `TORCH_CUDA_ARCH_LIST=8.6`.

### PowerShell vs CMD
All commands use PowerShell. Key differences:
- Activate venv: `.\venvbp\Scripts\activate` (not `venvbp\Scripts\activate` — needs `.\`)
- Environment vars: `$env:VAR="value"` (not `set VAR=value`)

### Long paths
Must be enabled before cloning PyTorch (filenames in submodules exceed Windows default):
```powershell
git config --global core.longpaths true
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

### Building wheels (not develop installs)
Use `python setup.py bdist_wheel` — produces a `.whl` in `dist\` that can be cached and
reused. Do NOT use `python setup.py develop` — that's not cacheable.

After each build:
```powershell
Copy-Item dist\*.whl C:\Users\tony\venvs\wheels\
```

### opencv-python build flags
Must use forward slashes in CMAKE_ARGS paths — PowerShell strips backslashes when passing
to CMake. Also requires explicit Python3 variable names:

```powershell
$env:CMAKE_ARGS = "-DWITH_CUDA=OFF -DWITH_OPENCL=OFF -DWITH_IPP=ON -DBUILD_opencv_python3=ON -DBUILD_opencv_python2=OFF -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_DOCS=OFF -DOPENCV_EXTRA_MODULES_PATH=C:/Users/tony/venvs/scr/opencv-python/opencv_contrib/modules -DPYTHON3_EXECUTABLE=C:/Users/tony/venvs/venvbp/Scripts/python.exe -DPYTHON3_INCLUDE_PATH=C:/Users/tony/AppData/Local/Python/pythoncore-3.14-64/Include -DPYTHON3_LIBRARIES=C:/Users/tony/AppData/Local/Python/pythoncore-3.14-64/libs/python314.lib -DPYTHON3_NUMPY_INCLUDE_DIRS=C:/Users/tony/venvs/venvbp/Lib/site-packages/numpy/_core/include"
```

### xformers — distributed import patch
xformers imports distributed training infrastructure (`modpar_layers`, `seqpar`,
`sequence_parallel_fused_ops`) unconditionally in `ops/__init__.py`. Our torch build
doesn't include `torch._C._distributed_c10d` (not needed for inference). The patch is
baked into the source at `xformers/ops/__init__.py` before building the wheel — all three
imports are wrapped in `try/except (ImportError, ModuleNotFoundError): pass`.

This is correct behavior for a single-GPU inference setup. ProPainter never uses distributed ops.

### xformers — triton warning
xformers prints "A matching Triton is not available" at import. This was resolved by
installing `triton-windows==3.6.0.post26` which has cp314 wheels. If it reappears in a
fresh venv, `pip install triton-windows` resolves it.

### Most other deps already have cp314 wheels
numpy, scipy, pillow, safetensors, transformers, huggingface-hub, triton-windows — install normally via pip.
Source builds are scoped to: torch, opencv-python, xformers only (torchvision skipped).

---

## Model Weights

Location: `weights/` folder in repo root

| File | Status | Source |
|------|--------|--------|
| `ProPainter.pth` | ✅ Downloaded | github.com/sczhou/ProPainter/releases/tag/v0.1.0 |
| `recurrent_flow_completion.pth` | ✅ Downloaded | Same release |
| `raft-things.pth` | ✅ Downloaded | Same release (selectable via backbone combobox) |
| `sea-raft-M.pth` | ✅ Downloaded | HuggingFace: MemorySlices/Tartan-C-T-TSKH-spring540x960-M |
| `waft-downstream.pth` | ✅ Downloaded | Google Drive — WAFT repo readme, "downstream applications" checkpoint. WAFT-dav2 (waft-a1) only — DA2 weights are baked in; no twins downstream checkpoint exists |
| `i3d_rgb_imagenet.pt` | 🗑️ DELETE | Eval only, not needed |

---

## Optical Flow — WAFT

**Why WAFT over RAFT/SEA-RAFT:**
- Ranks #1 on Spring, Sintel, and KITTI benchmarks simultaneously
- Replaces cost volume with high-resolution warping — lower VRAM than cost-volume approaches
- 1.3–4.1x faster than methods with comparable accuracy
- Same Princeton lab (Yihan Wang / Jia Deng), same author as SEA-RAFT
- ICLR 2026 Oral
- Targets Python 3.12 natively (we're going 3.14, source build)

**Repo:** `https://github.com/princeton-vl/WAFT` — use `waftv2` branch

**Integration:** `model/modules/flow_comp_waft.py` wraps WAFT's `ViTWarpV8` model.
Uses a scoped `sys.modules` swap to isolate WAFT's bare-name `model.*` imports from
ProPainter's own `model` package — evicts, imports, then restores in a `finally` block.

**Confirmed downstream target:** WAFT-dav2 (waft-a1 / `ViTWarpV8`). Checkpoint inspection
confirmed `waft-downstream.pth` is a waft-a1/dav2 checkpoint — DA2 weights are baked directly
into it and load from the HuggingFace cache (`~/.cache/huggingface`) on first instantiation.
No twins downstream checkpoint exists; WAFT-twins path in `flow_comp_waft.py` is dead code
retained until one is released.

**Flow backbone selector (implemented):**
`tkinter_app.py` has a backbone combobox: WAFT-dav2 (default), WAFT-twins, RAFT, SEA-RAFT.
RAFT and SEA-RAFT are available for regression A/B comparison of flow quality vs VRAM usage.

---

## TurboQuant — KV Cache Compression

**Why TurboQuant:**
ProPainter's spatiotemporal transformer maintains keys and values across reference frames to
keep textures consistent over time. The sparse attention workaround in the original code
literally throws away context tokens to survive VRAM. This is a KV memory problem.

TurboQuant (Google Research, ICLR 2026) compresses KV vectors to 3–4 bits with provably
near-zero accuracy loss and zero calibration required.

**Implementation:** `model/modules/turboquant_kv.py` — Lloyd-Max quantization (precomputed
Gaussian codebooks at 2/3/4 bits) + QJL residual correction for keys. Hooked into
`SparseWindowAttention` via `use_turboquant` flag. Off by default — zero behavior change.

**Smoke test:** `python model/modules/turboquant_kv.py` — cosine similarity ≥ 0.9675 at 3-bit.

**Expected benefit:**
- Stop dropping tokens — retain full temporal context instead of sparse subset
- Wider reference windows (50–100 frames vs current sparse handful)
- More VRAM headroom for higher resolution processing

---

## Test Harness — tkinter_app.py

Three-tab Tkinter benchmarking harness. No web server, no temp files, direct file paths.

**Tab 1 — Run:**
- Video path + Mask path (Browse buttons, direct filesystem paths)
- Run label (e.g. "baseline", "tq-3bit") — output files never overwrite each other
- Settings: neighbor_length, ref_stride, subvideo_length, mask dilation (Spinbox)
- FP16 (ProPainter) toggle, FP16 WAFT toggle, TurboQuant toggle + bits Spinbox (enabled when TQ on)
- Flow backbone combobox: WAFT-dav2 (default), WAFT-twins, RAFT, SEA-RAFT
- Run button (inference in background thread, UI stays live)
- Open results/ button
- Scrolled log widget + ttk.Progressbar + live VRAM bar (allocated / total GB, updates every 500ms)
- Log emits: settings dump, device info (GPU name, CUDA version, PyTorch version, total VRAM), WAFT CPU offload status, per-chunk peak VRAM

**Tab 2 — Benchmark:**
- matplotlib line chart (FigureCanvasTkAgg) — X: elapsed time, Y: VRAM allocated (GB)
- One line per run label, colored, with legend
- Summary table: run label | total time | peak VRAM | avg time/chunk
- Export CSV button
- VRAM sampled every 500ms via root.after + after each chunk completes
- Chart and table refresh after every run

**Tab 3 — Compare (stub):**
- "Side-by-side video comparison — Phase 1.5"
- Full implementation deferred to Phase 1.5

**Output naming:** `results/{stem}_{run_label}_inpainted.mp4` and
`results/{stem}_{run_label}_comparison.mp4`

**Mask auto-detection:** extension-based — video exts → per-frame mask video
(`_masks_from_video`), otherwise → static image mask (`read_mask`).

---

## ProPainter Modernization Changes (Phase 1) — Status

| Change | Status |
|--------|--------|
| requirements.txt rewrite | ✅ |
| torchvision read_video → cv2 | ✅ |
| get_device() modernized (removed cudnn.is_available() guard) | ✅ |
| tvdcn wired into recurrent_flow_completion.py | ✅ |
| WAFT adapter (flow_comp_waft.py) | ✅ Complete |
| TurboQuant hook (sparse_transformer.py + turboquant_kv.py) | ✅ |
| Tkinter benchmarking harness | ✅ |
| Runtime bug fixes (settings passthrough, flow cache eviction, tensor cleanup) | ✅ |
| VRAM optimizations (inference_mode, WAFT CPU offload, 4K tiling, cudnn benchmark) | ✅ |

---

## Git Log (recent)

```
(pending commit) runtime bugs + VRAM optimizations + device info logging
(pending commit) Context.md / CLAUDE.md updates — WAFT confirmed dav2, backbone combobox, FP16 WAFT
c829fa7 cleanup: remove dead training/eval code and unused weights and fixed waft
80244d3 Update WAFT submodule: remove torchvision from inference path
590d30a Phase 1 complete: tkinter harness, merged Context.md, CLAUDE.md
44a992f gradio: copy-first upload + manual H.264 convert button
```

---

## Phase 2 Preview (Don't Build Yet)

- **Integration target:** OpenFX C++ plugin in DaVinci Resolve Studio
- **Architecture:** Unix socket communication between C++ plugin and persistent Python inference servers
- **Key principle:** Models stay resident in memory across frames — per-frame reload is a non-starter
- **Natron:** Was considered and abandoned
- **Additional tools on the horizon:** Automated background plate generation ("video archaeology"), Snapchat filter removal using identity-model-based reconstruction

---

## Longer-Term Hardware & Upgrade Notes

**Tesla P40 (24GB VRAM)** being considered — when it arrives:
- Swap flow backbone to `sea-raft-L` (not yet downloaded) or largest WAFT variant
- `sea-raft-M.pth` already on disk for reference

### DiffuEraser — P40 upgrade target
- **What:** Diffusion-based video inpainting; uses ProPainter as a structured prior/initialization, then applies a video diffusion U-Net on top.
- **Why relevant:** Explicitly outperforms ProPainter on temporal consistency benchmarks (DAVIS, YouTube-VOS). The diffusion pass smooths temporal flickering that ProPainter's propagation+transformer approach leaves behind.
- **Weights:** ~30 GB (SD-based video diffusion backbone). Not viable on RTX 3080 8 GB.
- **Action:** Revisit when P40 arrives. DiffuEraser would replace or wrap the current `InpaintGenerator` inference path.
- **Reference:** "DiffuEraser: Diffusion Model for Video Inpainting"

---

## Key Principles

- **Cleanup before modernization** — dead code removal precedes functional changes
- **Source builds are a deliberate commitment** — "drag these guys into the 2026s kicking and screaming"
- **Privacy is non-negotiable** — no cloud, no exceptions, DoD wipes after every job
- **Tkinter harness gates the node** — don't touch Phase 2 until Phase 1 is proven on real footage
- **TurboQuant is optional until validated** — A/B test quality before committing
- **Wheel cache pattern** — build once in `venvbp`, cache to `venvs\wheels\`, install from cache for every new node
- **torchvision is skipped** — only usage was `read_video`, replaced with cv2
- **--no-deps on custom wheels** — always install torch/opencv/xformers with --no-deps to prevent PyPI clobbering
- **Never --force-reinstall xformers** — pip will pull torch 2.11.0+cpu and nuke the custom wheel
- **Direct file paths** — no web server, no temp files, no Gradio; tkinter uses askopenfilename throughout

---

## Context Persistence

This `Context.md` should be committed to the repo root. Claude Code sessions load it
automatically via `CLAUDE.md` (which can symlink or duplicate key sections). Update it
when decisions change or new phases begin.
