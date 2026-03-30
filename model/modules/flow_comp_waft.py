import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add WAFT to path
_WAFT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..', 'WAFT')
sys.path.insert(0, os.path.abspath(_WAFT_ROOT))
sys.path.insert(0, os.path.abspath(os.path.join(_WAFT_ROOT, 'thirdparty', 'DepthAnythingV2')))

from model import fetch_model
from utils.utils import load_ckpt
from inference_tools import InferenceWrapper


class WAFT_bi(nn.Module):
    def __init__(self, model_path='weights/waft-downstream.pth', device='cuda'):
        super().__init__()

        # Build config namespace directly from WAFT/config/a1/tar-c-t.json values.
        # Not parsed at runtime — hardcoded to avoid argparse/JSON dependency at inference time.
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

        model = fetch_model(args)
        load_ckpt(model, model_path)
        model.to(device)
        model.eval()

        for p in model.parameters():
            p.requires_grad = False

        self.wrapper = InferenceWrapper(
            model,
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

            gt_flows_forward = torch.stack(forward_flows, dim=1)   # [b, l_t-1, 2, h, w]
            gt_flows_backward = torch.stack(backward_flows, dim=1)  # [b, l_t-1, 2, h, w]

        return gt_flows_forward, gt_flows_backward
