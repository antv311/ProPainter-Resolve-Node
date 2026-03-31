# CLAUDE.md — ProPainter-Resolve-Node
Last updated: 2026-03-30

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

### Phase 1 — Get the AI Guts Working (CURRENT)
Get the inference stack fully modernized and running on Python 3.14 + CUDA 13.1 + WAFT + TurboQuant. Validate everything through a **Gradio test interface** — not the Resolve node. The node comes later.

Deliverables:
- Python 3.14 venv with source-built PyTorch (CUDA 13.1) ✅ Wheel built & cached
- torchvision ✅ ELIMINATED — deform_conv2d replaced with tvdcn 1.1.0; read_video replaced with cv2
- opencv-python source build ✅ Wheel built & cached
- xformers source build ✅ Wheel built & cached
- WAFT replacing RAFT as the optical flow backbone ✅ Code complete (waft-downstream.pth checkpoint still needed)
- TurboQuant integrated into ProPainter's spatiotemporal attention layers ✅ Complete (cosine sim 0.9675)
- Gradio UI for end-to-end testing of inpainting on real client footage types ✅ Complete
- All dead training/eval code removed from the repo ⬜ Pending

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
| gradio | 6.10.0 |
| imageio | 2.37.3 |
| imageio-ffmpeg | 0.6.0 |

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
pip install triton-windows scipy pillow safetensors transformers huggingface-hub gradio imageio imageio-ffmpeg
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
| `raft-things.pth` | ✅ Downloaded | Same release (kept for reference, WAFT replaces it) |
| `sea-raft-M.pth` | ✅ Downloaded | HuggingFace: MemorySlices/Tartan-C-T-TSKH-spring540x960-M |
| `waft-downstream.pth` | ⬜ Pending | Google Drive — WAFT repo readme, "downstream applications" checkpoint |
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

**Integration work required:**
ProPainter currently loads RAFT via `model/modules/flow_comp_raft.py`. We replace this with
WAFT's inference path. WAFT is architecturally cleaner to integrate than SEA-RAFT because it
dropped the cost volume entirely — fewer moving parts in the adapter. WAFT returns flow only
(no uncertainty output unlike SEA-RAFT).

---

## TurboQuant — KV Cache Compression

**Why TurboQuant:**
ProPainter's spatiotemporal transformer maintains keys and values across reference frames to
keep textures consistent over time. The sparse attention workaround in the original code
literally throws away context tokens to survive VRAM. This is a KV memory problem.

TurboQuant (Google Research, ICLR 2026) compresses KV vectors to 3–4 bits with provably
near-zero accuracy loss and zero calibration required. Community PyTorch implementations
already exist.

**The adaptation:** Community implementations wrap HuggingFace's `past_key_values` interface
for autoregressive LLMs. ProPainter's attention is bidirectional (past AND future frames).
We hook directly into `model/modules/sparse_transformer.py` and compress the cross-frame
keys and values before they hit `F.scaled_dot_product_attention`.

**Expected benefit:**
- Stop dropping tokens — retain full temporal context instead of sparse subset
- Wider reference windows (50–100 frames vs current sparse handful)
- More VRAM headroom for higher resolution processing

**Implementation approach:** Add as an optional flag — enable/disable for A/B quality testing.
Don't bake it in unconditionally until we've validated output quality on real footage.

**Community code:**
- `pip install turboquant` (PyPI, Apache 2.0)
- `github.com/tonbistudio/turboquant-pytorch` — PyTorch from-scratch, 99.5% attention fidelity at 3-bit
- `github.com/OnlyTerp/turboquant` — reference implementation, pure PyTorch
- Google's official code expected Q2 2026

---

## Dead Code to Remove (Do First)

Before any functional changes, strip everything not needed for inference:

```
datasets/
train.py
configs/
scripts/compute_flow.py
scripts/evaluate_flow_completion.py
scripts/evaluate_propainter.py
weights/i3d_rgb_imagenet.pt
web_demos/          (or gradio_app.py / app.py — whatever the demo is called)
environment.yaml    (targets Python 3.8, replace with our own setup docs)
assets/             (GIFs and screenshots for README)
```

Keep: `model/`, `core/`, `utils/`, `inputs/` (test data), `inference_propainter.py`, `requirements.txt` (will be rewritten)

---

## ProPainter Modernization Changes (Phase 1)

Six code changes needed after dead code removal:

**1. requirements.txt** — Full rewrite targeting Python 3.14, PyTorch 2.12+, CUDA 13.1

**2. torchvision `read_video` replacement** — Replace `torchvision.io.read_video` with
`cv2.VideoCapture`. This is the ONLY torchvision usage in the codebase. Eliminates
the torchvision dependency entirely.

**3. `get_device()` in `model/misc.py`** — Old pattern using deprecated torch device
detection. Replace with modern `torch.cuda.is_available()` pattern.

**4. scipy mask memory chunking** — `scipy.ndimage.binary_dilation` on the full mask stack
is the primary OOM source on 8GB VRAM. Chunk it. Process N frames at a time, reassemble.
This is the most important fix for the 3080.

**5. WAFT adapter** — Replace `model/modules/flow_comp_raft.py` with WAFT inference path.

**6. TurboQuant hook** — Wrap attention in `model/modules/sparse_transformer.py` with
optional TurboQuant compression.

---

## Gradio Test Interface (Phase 1 Validation)

Replace the existing Gradio demo (which we're deleting) with a purpose-built test UI that
matches our actual use cases:

- **Input:** Video file + mask (drawn or uploaded)
- **Controls:** Neighbor length, ref stride, subvideo length, fp16 toggle, TurboQuant on/off, VRAM usage display
- **Output:** Inpainted video + side-by-side comparison
- **Logging:** Frame processing time, peak VRAM per chunk

This is our quality gate before Phase 2.

---

## Phase 2 Preview (Don't Build Yet)

- **Integration target:** OpenFX C++ plugin in DaVinci Resolve Studio
- **Architecture:** Unix socket communication between C++ plugin and persistent Python inference servers
- **Key principle:** Models stay resident in memory across frames — per-frame reload is a non-starter
- **Natron:** Was considered and abandoned
- **Additional tools on the horizon:** Automated background plate generation ("video archaeology"), Snapchat filter removal using identity-model-based reconstruction

---

## Longer-Term Hardware

Tesla P40 (24GB VRAM) being considered — when it arrives, swap to `sea-raft-L.pth` and the
largest WAFT variant. The SEA-RAFT M model (`sea-raft-M.pth`) is already downloaded and
ready for reference.

---

## Key Principles

- **Cleanup before modernization** — dead code removal precedes functional changes
- **Source builds are a deliberate commitment** — "drag these guys into the 2026s kicking and screaming"
- **Privacy is non-negotiable** — no cloud, no exceptions, DoD wipes after every job
- **Gradio gates the node** — don't touch Phase 2 until Phase 1 is proven on real footage
- **TurboQuant is optional until validated** — A/B test quality before committing
- **Wheel cache pattern** — build once in `venvbp`, cache to `venvs\wheels\`, install from cache for every new node
- **torchvision is skipped** — only usage was `read_video`, replaced with cv2
- **--no-deps on custom wheels** — always install torch/opencv/xformers with --no-deps to prevent PyPI clobbering
- **Never --force-reinstall xformers** — pip will pull torch 2.11.0+cpu and nuke the custom wheel

---

## Context Persistence

This `CLAUDE.md` should be committed to the repo root. Claude Code sessions load it
automatically. Update it when decisions change or new phases begin.
