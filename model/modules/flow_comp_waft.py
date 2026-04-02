import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

# Absolute path to the WAFT submodule — resolved once at import time.
_WAFT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'WAFT'))
_WAFT_DAV2 = os.path.join(_WAFT_ROOT, 'thirdparty', 'DepthAnythingV2')

# ── scoped-import helper ───────────────────────────────────────────────────────
# WAFT's model.* uses bare absolute names that collide with ProPainter's own
# `model` package.  This context manager evicts ProPainter's entries, puts
# WAFT on sys.path, does the import, then restores everything.

from contextlib import contextmanager

@contextmanager
def _waft_import_scope():
    saved = {k: sys.modules.pop(k)
             for k in list(sys.modules)
             if k == 'model' or k.startswith('model.')}
    prev_path = sys.path[:]
    sys.path.insert(0, _WAFT_DAV2)
    sys.path.insert(0, _WAFT_ROOT)
    try:
        yield
    finally:
        for k in [k for k in sys.modules if k == 'model' or k.startswith('model.')]:
            del sys.modules[k]
        sys.modules.update(saved)
        sys.path[:] = prev_path


# ── WAFT_bi: unified flow model wrapper ───────────────────────────────────────

class WAFT_bi(nn.Module):
    """
    Unified optical-flow wrapper.  backbone selects one of:
      'WAFT-dav2'   — ViTWarpV8 (waft-a1) with DepthAnythingV2 feature extractor.
                      Requires depth-anything-ckpts/depth_anything_v2_vits.pth.
      'WAFT-twins'  — WAFTv2 (waft-a2) with TwinsFeatureEncoder.
                      No DepthAnything checkpoint needed.
      'RAFT'        — Classic RAFT-Things.
      'SEA-RAFT'    — SEA-RAFT-M.  (SEA-RAFT code submodule must be present.)

    All four share the same call interface:
        flow_fwd, flow_bwd = model(frames_t, iters=20)
    where frames_t is [B, T, C, H, W] in [-1, 1].
    """

    def __init__(self, model_path: str, device='cuda', backbone: str = 'WAFT-twins'):
        super().__init__()
        self.backbone = backbone

        if backbone == 'WAFT-dav2':
            self._init_waft_dav2(model_path, device)
        elif backbone == 'WAFT-twins':
            self._init_waft_twins(model_path, device)
        elif backbone == 'RAFT':
            self._init_raft(model_path, device)
        elif backbone == 'SEA-RAFT':
            self._init_searaft(model_path, device)
        else:
            raise ValueError(f"Unknown backbone '{backbone}'. "
                             "Choose from: WAFT-dav2, WAFT-twins, RAFT, SEA-RAFT")

        self.eval()

    # ── WAFT-dav2 ─────────────────────────────────────────────────────────────

    def _init_waft_dav2(self, model_path, device):
        args = argparse.Namespace(
            name='tar-c-t',
            dataset='things',
            dav2_backbone='vits',
            network_backbone='dav2',   # triggers DepthAnything load in ViTWarpV8
            algorithm='waft-a1',
            use_var=True,
            var_min=0,
            var_max=10,
            iters=5,
            image_size=[432, 960],
            scale=0,
            dropout=0,
        )
        with _waft_import_scope():
            from model.waft_a1 import ViTWarpV8

        self._waft = ViTWarpV8(args)
        sd = torch.load(model_path, map_location='cpu')
        self._waft.load_state_dict(sd, strict=False)
        self._waft.to(device).eval()
        for p in self._waft.parameters():
            p.requires_grad = False

    # ── WAFT-twins ────────────────────────────────────────────────────────────

    def _init_waft_twins(self, model_path, device):
        args = argparse.Namespace(
            name='tar-c-t',
            dataset='things',
            feature_encoder='twins',
            iterative_module='vits',
            algorithm='waft-a2',
            use_var=True,
            var_min=0,
            var_max=10,
            iters=5,
            image_size=[432, 960],
            scale=0,
            dropout=0,
        )
        with _waft_import_scope():
            from model.waft_a2 import WAFTv2
            from inference_tools import InferenceWrapper

        waft_model = WAFTv2(args)
        state_dict = torch.load(model_path, map_location='cpu')
        waft_model.load_state_dict(state_dict, strict=False)
        waft_model.to(device).eval()
        for p in waft_model.parameters():
            p.requires_grad = False

        self.wrapper = InferenceWrapper(
            waft_model, scale=0, train_size=None,
            pad_to_train_size=False, tiling=False,
        )

    # ── RAFT ──────────────────────────────────────────────────────────────────

    def _init_raft(self, model_path, device):
        from model.modules.flow_comp_raft import RAFT_bi
        raft = RAFT_bi(model_path=model_path, device=device)
        for p in raft.parameters():
            p.requires_grad = False
        self._raft = raft

    # ── SEA-RAFT ──────────────────────────────────────────────────────────────

    def _init_searaft(self, model_path, device):
        # SEA-RAFT requires a separate submodule / package not yet integrated.
        # This will raise a clear error at load time if the code is absent.
        try:
            from sea_raft.model import SEARaft  # placeholder — update if submodule added
        except ImportError as exc:
            raise ImportError(
                "SEA-RAFT Python package not found. "
                "Clone https://github.com/princeton-vl/SEA-RAFT and add it to "
                "sys.path, or install it, before selecting the SEA-RAFT backbone."
            ) from exc
        searaft = SEARaft()
        state_dict = torch.load(model_path, map_location='cpu')
        searaft.load_state_dict(state_dict, strict=False)
        searaft.to(device).eval()
        for p in searaft.parameters():
            p.requires_grad = False
        self._searaft = searaft

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, gt_local_frames, iters=20):
        """
        gt_local_frames: [B, T, C, H, W] in [-1, 1]
        Returns (gt_flows_forward, gt_flows_backward) each [B, T-1, 2, H, W]
        """
        with torch.no_grad():
            b, l_t, c, h, w = gt_local_frames.size()

            if self.backbone == 'WAFT-dav2':
                # ViTWarpV8.forward() expects [0, 255] and handles padding internally.
                # Output dict: {'flow': [pred_iter0, ..., pred_iterN], 'info': [...]}
                # Take ['flow'][-1] — the final refined estimate.
                frames_255 = ((gt_local_frames + 1) * 127.5).clamp(0, 255)
                forward_flows, backward_flows = [], []
                for i in range(l_t - 1):
                    img1 = frames_255[:, i]
                    img2 = frames_255[:, i + 1]
                    forward_flows.append(self._waft(img1, img2)['flow'][-1])
                    backward_flows.append(self._waft(img2, img1)['flow'][-1])
                return (
                    torch.stack(forward_flows,  dim=1),
                    torch.stack(backward_flows, dim=1),
                )

            elif self.backbone == 'WAFT-twins':
                # Dead code until a twins downstream checkpoint is released.
                frames_255 = ((gt_local_frames + 1) * 127.5).clamp(0, 255)
                forward_flows, backward_flows = [], []
                for i in range(l_t - 1):
                    img1 = frames_255[:, i]
                    img2 = frames_255[:, i + 1]
                    forward_flows.append(self.wrapper.calc_flow(img1, img2)['flow'][-1])
                    backward_flows.append(self.wrapper.calc_flow(img2, img1)['flow'][-1])
                return (
                    torch.stack(forward_flows,  dim=1),
                    torch.stack(backward_flows, dim=1),
                )

            elif self.backbone == 'RAFT':
                return self._raft(gt_local_frames, iters=iters)

            elif self.backbone == 'SEA-RAFT':
                # SEA-RAFT shares the RAFT-style forward interface
                frames_255 = ((gt_local_frames + 1) * 127.5).clamp(0, 255)
                gtlf_1 = frames_255[:, :-1].reshape(-1, c, h, w)
                gtlf_2 = frames_255[:, 1: ].reshape(-1, c, h, w)
                _, fwd = self._searaft(gtlf_1, gtlf_2, iters=iters, test_mode=True)
                _, bwd = self._searaft(gtlf_2, gtlf_1, iters=iters, test_mode=True)
                return (
                    fwd.view(b, l_t - 1, 2, h, w),
                    bwd.view(b, l_t - 1, 2, h, w),
                )
