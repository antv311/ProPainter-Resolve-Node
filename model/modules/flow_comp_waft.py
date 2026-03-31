import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

# Absolute path to the WAFT submodule — resolved once at import time.
_WAFT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'WAFT'))
_WAFT_DAV2 = os.path.join(_WAFT_ROOT, 'thirdparty', 'DepthAnythingV2')


class WAFT_bi(nn.Module):
    def __init__(self, model_path='weights/waft-downstream.pth', device='cuda'):
        super().__init__()

        # Build config namespace directly from WAFT/config/a1/tar-c-t.json values.
        # Hardcoded — no argparse or JSON parsing at runtime.
        args = argparse.Namespace(
            name='tar-c-t',
            dataset='things',
            dav2_backbone='vits',
            network_backbone='vits',
            algorithm='waft-a1',
            use_var=True,
            var_min=0,
            var_max=10,
            iters=5,
            image_size=[432, 960],
            scale=0,
            dropout=0,
        )

        # ── scoped import: isolate WAFT's `model.*` from ProPainter's ──────────
        # WAFT/model/waft_a1.py does `from model.backbone.xxx import ...` using
        # bare absolute names that collide with ProPainter's `model` package.
        # We temporarily evict ProPainter's model entries from sys.modules, put
        # WAFT_ROOT on the front of sys.path, do the WAFT imports, then restore.
        _saved = {k: sys.modules.pop(k)
                  for k in list(sys.modules)
                  if k == 'model' or k.startswith('model.')}
        _prev_path = sys.path[:]
        sys.path.insert(0, _WAFT_DAV2)
        sys.path.insert(0, _WAFT_ROOT)

        try:
            from model.waft_a1 import ViTWarpV8       # WAFT's model, not ProPainter's
            from inference_tools import InferenceWrapper
        finally:
            # Purge WAFT's model entries, restore ProPainter's.
            for k in [k for k in sys.modules if k == 'model' or k.startswith('model.')]:
                del sys.modules[k]
            sys.modules.update(_saved)
            sys.path[:] = _prev_path
        # ── end scoped import ─────────────────────────────────────────────────

        # `fetch_model` with algorithm='waft-a1' is just ViTWarpV8(args).
        # `load_ckpt` is: load state dict, load_state_dict(strict=False).
        waft_model = ViTWarpV8(args)
        state_dict = torch.load(model_path, map_location='cpu')
        waft_model.load_state_dict(state_dict, strict=False)
        waft_model.to(device).eval()

        for p in waft_model.parameters():
            p.requires_grad = False

        self.wrapper = InferenceWrapper(
            waft_model,
            scale=0,
            train_size=None,
            pad_to_train_size=False,
            tiling=False,
        )
        self.eval()

    def forward(self, gt_local_frames, iters=20):
        # iters kept for call-site compatibility; WAFT's InferenceWrapper handles iteration.
        with torch.no_grad():
            b, l_t, c, h, w = gt_local_frames.size()

            # Convert from [-1, 1] to [0, 255]
            frames_255 = ((gt_local_frames + 1) * 127.5).clamp(0, 255)

            forward_flows = []
            backward_flows = []

            for i in range(l_t - 1):
                img1 = frames_255[:, i]      # [b, c, h, w]
                img2 = frames_255[:, i + 1]  # [b, c, h, w]

                fwd = self.wrapper.calc_flow(img1, img2)
                forward_flows.append(fwd['flow'][-1])  # [b, 2, h, w]

                bwd = self.wrapper.calc_flow(img2, img1)
                backward_flows.append(bwd['flow'][-1])  # [b, 2, h, w]

            gt_flows_forward  = torch.stack(forward_flows,  dim=1)  # [b, l_t-1, 2, h, w]
            gt_flows_backward = torch.stack(backward_flows, dim=1)  # [b, l_t-1, 2, h, w]

        return gt_flows_forward, gt_flows_backward
