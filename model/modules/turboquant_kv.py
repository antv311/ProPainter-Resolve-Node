"""
TurboQuant KV cache compression for ProPainter's SparseWindowAttention.

Algorithm (TurboQuant, Google Research ICLR 2026):
  Keys   — Stage 1: Randomized Hadamard Transform + Lloyd-Max scalar quantization (2/3/4-bit)
            Stage 2: 1-bit QJL residual correction for unbiased inner-product estimation
  Values — Stage 1 only (MSE-optimal compression; no inner-product bias needed for weighted sum)

All codebooks are precomputed from standard Gaussian N(0,1) — no calibration data required.
All operations stay on the CUDA device; no CPU round-trips.
No LAPACK dependency: rotation uses a Randomized Hadamard Transform (RHT) with stored ±1 signs.
"""

import math
from dataclasses import dataclass
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Lloyd-Max codebooks for standard Gaussian N(0,1)
# Boundaries: inner decision thresholds (sorted ascending, symmetric).
# Centroids:  reconstruction values for each quantisation bin.
# From Max (1960) / standard optimal scalar quantisation tables.
# ---------------------------------------------------------------------------
_LLOYD_MAX = {
    2: {
        # 4 levels, 3 inner boundaries
        "boundaries": torch.tensor([-0.9674, 0.0000, 0.9674]),
        "centroids":  torch.tensor([-1.5104, -0.4528, 0.4528, 1.5104]),
    },
    3: {
        # 8 levels, 7 inner boundaries
        "boundaries": torch.tensor([-1.7480, -1.0500, -0.5006, 0.0000,
                                     0.5006,  1.0500,  1.7480]),
        "centroids":  torch.tensor([-2.1518, -1.3439, -0.7560, -0.2451,
                                     0.2451,  0.7560,  1.3439,  2.1518]),
    },
    4: {
        # 16 levels, 15 inner boundaries
        "boundaries": torch.tensor([
            -2.4008, -1.8435, -1.4372, -1.0993, -0.7996,
            -0.5224, -0.2582,  0.0000,  0.2582,  0.5224,
             0.7996,  1.0993,  1.4372,  1.8435,  2.4008,
        ]),
        "centroids": torch.tensor([
            -2.7326, -2.0690, -1.6180, -1.2562, -0.9424,
            -0.6568, -0.3880, -0.1284,  0.1284,  0.3880,
             0.6568,  0.9424,  1.2562,  1.6180,  2.0690,
             2.7326,
        ]),
    },
}


# ---------------------------------------------------------------------------
# Fast Walsh-Hadamard Transform (FWHT) — pure PyTorch, no LAPACK
# ---------------------------------------------------------------------------

def _fwht(x: torch.Tensor) -> torch.Tensor:
    """
    Unnormalized Fast Walsh-Hadamard Transform on the last dimension.
    n = x.shape[-1] must be a power of 2.
    """
    n = x.shape[-1]
    h = 1
    while h < n:
        xv = x.reshape(*x.shape[:-1], -1, 2 * h)   # [..., n//(2h), 2h]
        a = xv[..., :h]                              # [..., n//(2h), h]
        b = xv[..., h:]
        x = torch.cat([a + b, a - b], dim=-1).reshape(x.shape)
        h *= 2
    return x


# ---------------------------------------------------------------------------
# Compressed representation types
# ---------------------------------------------------------------------------

@dataclass
class CompressedKeys:
    codes:     torch.Tensor  # uint8  [..., c_head]   — Lloyd-Max bin indices
    scales:    torch.Tensor  # fp16   [..., 1]         — per-vector scale (norm/sqrt(c_head))
    qjl_signs: torch.Tensor  # int8   [..., jl_dim]   — sign(normalised_residual @ J)
    qjl_norms: torch.Tensor  # fp16   [..., 1]         — L2 norm of quantisation residual
    m:         int                                      # JL projection dimension


@dataclass
class CompressedValues:
    codes:  torch.Tensor  # uint8  [..., c_head]
    scales: torch.Tensor  # fp16   [..., 1]


@dataclass
class CompressedKV:
    anchor_k:     Optional[torch.Tensor]        # fp16 — anchor frame key tokens
    anchor_v:     Optional[torch.Tensor]        # fp16 — anchor frame value tokens
    compressed_k: Optional[CompressedKeys]      # compressed non-anchor keys
    compressed_v: Optional[CompressedValues]    # compressed non-anchor values
    anchor_mask:  Optional[torch.Tensor]        # bool [N_tokens]
    N_tokens:     int


# ---------------------------------------------------------------------------
# TurboQuantCompressor
# ---------------------------------------------------------------------------

class TurboQuantCompressor(nn.Module):
    """
    Implements TurboQuant two-stage compression for a single attention head.

    Rotation is implemented as a Randomized Hadamard Transform (RHT):
        rht(x)  = FWHT(x * R_signs) / sqrt(c_head)   — orthogonal forward transform
        irht(y) = FWHT(y) * R_signs / sqrt(c_head)   — orthogonal inverse transform

    This requires c_head to be a power of 2 (true for ProPainter: 128 = 2^7).

    Registered buffers (follow .to(device) automatically):
      R_signs   — Rademacher ±1 signs for the RHT [c_head]
      J         — Rademacher JL projection matrix  [c_head, jl_dim]
      boundaries — Lloyd-Max decision thresholds
      centroids  — Lloyd-Max reconstruction values
    """

    def __init__(self, c_head: int, bits: int = 3, jl_dim: Optional[int] = None):
        super().__init__()
        assert bits in _LLOYD_MAX, f"bits must be one of {list(_LLOYD_MAX.keys())}, got {bits}"
        assert c_head > 0 and (c_head & (c_head - 1)) == 0, \
            f"c_head must be a power of 2, got {c_head}"
        self.c_head = c_head
        self.bits = bits
        self.jl_dim = jl_dim if jl_dim is not None else c_head

        # RHT: random ±1 signs (no LAPACK needed)
        R_signs = (torch.randint(0, 2, (c_head,)) * 2 - 1).float()
        self.register_buffer("R_signs", R_signs)  # [c_head]

        # Rademacher JL projection: entries ∈ {±1/√jl_dim}
        J = (torch.randint(0, 2, (c_head, self.jl_dim)) * 2 - 1).float()
        J = J / math.sqrt(self.jl_dim)
        self.register_buffer("J", J)  # [c_head, jl_dim]

        cb = _LLOYD_MAX[bits]
        self.register_buffer("boundaries", cb["boundaries"])  # [2^bits - 1]
        self.register_buffer("centroids",  cb["centroids"])   # [2^bits]

    # ------------------------------------------------------------------
    # RHT helpers — operate on the last dimension
    # ------------------------------------------------------------------

    def _rht(self, x: torch.Tensor) -> torch.Tensor:
        """Forward RHT: x → FWHT(x * R_signs) / sqrt(c_head)."""
        return _fwht(x * self.R_signs) / math.sqrt(self.c_head)

    def _irht(self, y: torch.Tensor) -> torch.Tensor:
        """Inverse RHT: y → FWHT(y) * R_signs / sqrt(c_head)."""
        return _fwht(y) * self.R_signs / math.sqrt(self.c_head)

    # ------------------------------------------------------------------
    # Internal quantisation helpers
    # ------------------------------------------------------------------

    def _quantize(self, x_rot: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_rot:  [..., c_head]  float32 — rotated, un-normalised vectors
            scales: [..., 1]       float32 — per-vector scale = norm / sqrt(c_head)
        Returns:
            codes: uint8 [..., c_head]
        """
        x_norm = x_rot / (scales + 1e-8)
        # bucketize → indices in [0, 2^bits - 1]
        codes = torch.bucketize(x_norm.contiguous(), self.boundaries)
        return codes.to(torch.uint8)

    def _dequantize(self, codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        """
        Args:
            codes:  uint8  [..., c_head]
            scales: float32 [..., 1]
        Returns:
            float32 [..., c_head]
        """
        recon = self.centroids[codes.long()]  # [..., c_head]
        return recon * scales

    # ------------------------------------------------------------------
    # Public compression API
    # ------------------------------------------------------------------

    def compress_keys(self, k: torch.Tensor) -> CompressedKeys:
        """
        Args:
            k: [..., c_head]  (any float dtype)
        Returns: CompressedKeys
        """
        k_rot = self._rht(k.float())                                    # [..., c_head]
        # Scale: norm / sqrt(c_head) normalises each coordinate to N(0,1)
        scales = (k_rot.norm(dim=-1, keepdim=True) / math.sqrt(self.c_head))
        codes = self._quantize(k_rot, scales)                           # uint8

        # Quantisation residual for QJL stage
        k_recon = self._dequantize(codes, scales)                       # [..., c_head]
        residual = k_rot - k_recon                                      # [..., c_head]
        qjl_norms = residual.norm(dim=-1, keepdim=True)                 # [..., 1]

        # Normalise residual then project; sign gives 1-bit QJL codes
        residual_norm = residual / (qjl_norms + 1e-8)                  # [..., c_head]
        proj = residual_norm @ self.J                                   # [..., jl_dim]
        qjl_signs = proj.sign().to(torch.int8)                          # ±1 as int8

        del k_rot, k_recon, residual, residual_norm, proj
        return CompressedKeys(
            codes=codes,
            scales=scales.half(),
            qjl_signs=qjl_signs,
            qjl_norms=qjl_norms.half(),
            m=self.jl_dim,
        )

    def compress_values(self, v: torch.Tensor) -> CompressedValues:
        """
        Args:
            v: [..., c_head]  (any float dtype)
        Returns: CompressedValues
        """
        v_rot = self._rht(v.float())                                    # [..., c_head]
        scales = v_rot.norm(dim=-1, keepdim=True) / math.sqrt(self.c_head)
        codes = self._quantize(v_rot, scales)                           # uint8
        del v_rot
        return CompressedValues(codes=codes, scales=scales.half())

    def asymmetric_attention_scores(
        self,
        q: torch.Tensor,
        compressed_k: CompressedKeys,
    ) -> torch.Tensor:
        """
        Compute q @ k.T ≈ scores without fully decompressing k.

        Stage 1: rotate q into key-space, matmul with dequantised k_rot.
        Stage 2: add QJL inner-product correction for the quantisation residual.

        Inner products are preserved under RHT: rht(q) · rht(k) = q · k,
        so we can work entirely in rotated space.

        Args:
            q:            [..., N_q, c_head]
            compressed_k: CompressedKeys with shapes [..., N_k, ...]
        Returns:
            scores: [..., N_q, N_k]  (same dtype as q)
        """
        q_rot = self._rht(q.float())                                    # [..., N_q, c_head]

        # Stage 1 — dequantise key approximation and compute dot products
        k_approx_rot = self._dequantize(
            compressed_k.codes,
            compressed_k.scales.float(),
        )                                                               # [..., N_k, c_head]
        scores = q_rot @ k_approx_rot.transpose(-2, -1)                # [..., N_q, N_k]

        # Stage 2 — QJL residual correction (unbiased inner-product estimate of residual)
        # E[Σ_d sign((J^T r_norm)_d) * (J^T q)_d] = sqrt(jl_dim) * sqrt(2/π) * (r_norm · q)
        # so the unbiased estimate requires dividing by that factor.
        q_proj = q_rot @ self.J                                        # [..., N_q, jl_dim]
        qjl_s  = compressed_k.qjl_signs.float()                       # [..., N_k, jl_dim]
        qjl_n  = compressed_k.qjl_norms.float()                       # [..., N_k, 1]
        qjl_bias = math.sqrt(self.jl_dim) * math.sqrt(2.0 / math.pi)  # bias correction factor
        # correction[..., q_i, k_j] ≈ r_kj · q_roti  (residual inner product estimate)
        correction = (q_proj @ qjl_s.transpose(-2, -1)) * qjl_n.squeeze(-1).unsqueeze(-2)
        correction = correction / qjl_bias
        scores = scores + correction

        return scores.to(q.dtype)

    def decompress_values(
        self,
        compressed_v: CompressedValues,
        target_dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Reconstruct value vectors by dequantising and inverse-rotating.

        Args:
            compressed_v: CompressedValues
            target_dtype: desired output dtype
        Returns:
            [..., c_head] in target_dtype
        """
        v_rot = self._dequantize(compressed_v.codes, compressed_v.scales.float())
        return self._irht(v_rot).to(target_dtype)


# ---------------------------------------------------------------------------
# BidirectionalTQKVCache
# ---------------------------------------------------------------------------

class BidirectionalTQKVCache(nn.Module):
    """
    Wraps TurboQuantCompressor for ProPainter's SparseWindowAttention.

    Anchor tokens (identified by anchor_mask) stay in FP16.
    Non-anchor tokens are compressed with TurboQuant.
    """

    def __init__(self, c_head: int, bits: int = 3):
        super().__init__()
        self.compressor = TurboQuantCompressor(c_head=c_head, bits=bits)

    def compress(
        self,
        win_k_t: torch.Tensor,
        win_v_t: torch.Tensor,
        anchor_mask: Optional[torch.Tensor] = None,
    ) -> CompressedKV:
        """
        Compress KV tensors, keeping anchor tokens in FP16.

        Args:
            win_k_t:     [mask_n, n_head, N_tokens, c_head]
            win_v_t:     [mask_n, n_head, N_tokens, c_head]
            anchor_mask: bool [N_tokens] — True positions kept as FP16
        Returns:
            CompressedKV
        """
        N_tokens = win_k_t.shape[2]
        has_anchors = (
            anchor_mask is not None
            and anchor_mask.any().item()
            and (~anchor_mask).any().item()
        )

        if not has_anchors:
            # Compress everything
            ck = self.compressor.compress_keys(win_k_t)
            cv = self.compressor.compress_values(win_v_t)
            return CompressedKV(
                anchor_k=None, anchor_v=None,
                compressed_k=ck, compressed_v=cv,
                anchor_mask=None, N_tokens=N_tokens,
            )

        non_anchor = ~anchor_mask
        anchor_k = win_k_t[:, :, anchor_mask, :]   # [mask_n, n_head, A, c_head] fp16
        anchor_v = win_v_t[:, :, anchor_mask, :]
        na_k = win_k_t[:, :, non_anchor, :]        # [mask_n, n_head, NA, c_head]
        na_v = win_v_t[:, :, non_anchor, :]

        ck = self.compressor.compress_keys(na_k)
        cv = self.compressor.compress_values(na_v)

        return CompressedKV(
            anchor_k=anchor_k, anchor_v=anchor_v,
            compressed_k=ck, compressed_v=cv,
            anchor_mask=anchor_mask, N_tokens=N_tokens,
        )

    def compute_attention(
        self,
        win_q_t: torch.Tensor,
        compressed_kv: CompressedKV,
        scale: float,
        dropout_fn: Optional[Callable] = None,
    ) -> torch.Tensor:
        """
        Compute scaled dot-product attention over the compressed KV cache.

        Keys:   non-anchor via asymmetric inner product; anchor via direct matmul.
        Values: non-anchor decompressed just-in-time; anchor FP16 directly.

        Args:
            win_q_t:       [mask_n, n_head, N_q, c_head]
            compressed_kv: CompressedKV from self.compress()
            scale:         attention scale (typically 1/sqrt(c_head))
            dropout_fn:    optional attn_drop callable
        Returns:
            [mask_n, n_head, N_q, c_head]
        """
        score_parts = []
        value_parts = []

        if compressed_kv.compressed_k is not None:
            na_scores = self.compressor.asymmetric_attention_scores(
                win_q_t, compressed_kv.compressed_k
            )                                                           # [..., N_q, NA]
            na_v = self.compressor.decompress_values(
                compressed_kv.compressed_v, win_q_t.dtype
            )                                                           # [..., NA, c_head]
            score_parts.append(na_scores)
            value_parts.append(na_v)

        if compressed_kv.anchor_k is not None:
            anc_k = compressed_kv.anchor_k.to(win_q_t.dtype)
            anc_scores = win_q_t @ anc_k.transpose(-2, -1)             # [..., N_q, A]
            anc_v = compressed_kv.anchor_v.to(win_q_t.dtype)
            score_parts.append(anc_scores)
            value_parts.append(anc_v)

        all_scores = torch.cat(score_parts, dim=-1) * scale            # [..., N_q, N_k]
        all_v      = torch.cat(value_parts, dim=-2)                    # [..., N_k, c_head]

        att = F.softmax(all_scores, dim=-1)
        if dropout_fn is not None:
            att = dropout_fn(att)

        return att @ all_v                                              # [..., N_q, c_head]


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running smoke test on {device}...")

    # ProPainter actual dims: hidden=512, n_head=4 → c_head = 512 // 4 = 128
    c_head, bits, n_head, mask_n = 128, 3, 4, 4
    n_t, N_spatial = 10, 10
    N_tokens = n_t * N_spatial   # 100
    N_q = 45                     # w_h * w_w for ProPainter (5*9)
    scale = 1.0 / math.sqrt(c_head)

    torch.manual_seed(0)
    cache   = BidirectionalTQKVCache(c_head=c_head, bits=bits).to(device)
    win_k   = torch.randn(mask_n, n_head, N_tokens, c_head, device=device)
    win_v   = torch.randn(mask_n, n_head, N_tokens, c_head, device=device)
    win_q   = torch.randn(mask_n, n_head, N_q,      c_head, device=device)

    # --- Reference (uncompressed) ---
    att_ref = F.softmax(win_q @ win_k.transpose(-2, -1) * scale, dim=-1)
    y_ref   = att_ref @ win_v

    # --- Anchor mask: first and last temporal frame ---
    N_spatial_per_frame = N_tokens // n_t
    anchor_mask = torch.zeros(N_tokens, dtype=torch.bool, device=device)
    anchor_mask[:N_spatial_per_frame] = True                         # first frame
    anchor_mask[(n_t - 1) * N_spatial_per_frame:] = True            # last frame

    # --- TurboQuant path ---
    cKV  = cache.compress(win_k, win_v, anchor_mask)
    y_tq = cache.compute_attention(win_q, cKV, scale)

    assert y_tq.shape == y_ref.shape, f"Shape mismatch: {y_tq.shape} vs {y_ref.shape}"

    # Cosine similarity averaged over the batch
    cos = F.cosine_similarity(
        y_ref.reshape(mask_n, -1),
        y_tq.reshape(mask_n, -1),
        dim=-1,
    ).mean()
    print(f"Cosine similarity (TurboQuant vs reference): {cos.item():.4f}")

    threshold = 0.95
    if cos.item() <= threshold:
        print(f"FAIL — cosine similarity {cos.item():.4f} ≤ {threshold}", file=sys.stderr)
        sys.exit(1)

    print("Smoke test passed.")
