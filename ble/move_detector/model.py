"""
model.py

Onset detector: per-frame CNN encoder + temporal convolutional network,
producing one "a move is turning here" logit per frame.

Shape of the idea
-----------------
The old approach (live_test.py's MotionGate) thresholded a global
frame-delta and waited for stillness to close a move window. That fails
for two reasons this model is built to fix:

  1. During a solve the cube never returns to rest. Median inter-move gap
     in the recorded sessions is 420ms and p10 is 180ms, while the gate
     needed 133ms of confirmed stillness — so fast moves merged.
  2. A whole-cube rotation or a regrip produces a LARGER frame delta than
     a layer turn. Magnitude cannot separate them; only learned appearance
     can.

So instead of segmenting on stillness, this predicts a per-frame score
that peaks at the moment of a turn, and peaks are picked out of it
(decode.py). Supervision is free: every BLE move timestamp is a labeled
onset.

Architecture
------------
  FrameEncoder   (T, 2, 96, 96) -> (T, 128)     per-frame, weights shared
                 channel 0 = grayscale, channel 1 = diff from prev frame
  TCN            4 residual blocks, dilations 1/2/4/8, kernel 3
                 receptive field 61 frames ~= 2.0s at 30fps
  head           1x1 conv -> (T,) logits

Why a TCN rather than the ConvLSTM: with ~800 moves from a dozen
homogeneous sessions, a ConvLSTM over raw frames has enough capacity to
memorize grip and lighting, and you cannot then tell "wrong architecture"
apart from "not enough data". This is ~1.4M parameters, trains in minutes,
and gives a baseline to beat. Swapping the TCN for a ConvLSTM later is a
change to one module.

NON-CAUSAL by design: convolutions are centered, so frame t sees ~1s on
either side. That suits the intended flow — buffer the solve, then run
detection and classification offline afterwards (a 54s solve is only ~15MB
of 96x96 grayscale). For true streaming you would need causal padding,
which costs accuracy at the leading edge.
"""

import numpy as np
import torch
import torch.nn as nn


FRAME_SIZE  = 96     # encoder input, must match prepare_data.py
IN_CHANNELS = 2      # grayscale + temporal diff
FEAT_DIM    = 128    # per-frame embedding width
TCN_DILATIONS = (1, 2, 4, 8)


class FrameEncoder(nn.Module):
    """(N, 2, 96, 96) -> (N, FEAT_DIM). Applied to every frame independently."""

    def __init__(self, feat_dim: int = FEAT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(IN_CHANNELS, 32, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),       # 48x48
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),       # 24x24
            nn.Conv2d(64, 96, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96), nn.ReLU(inplace=True),       # 12x12
            nn.Conv2d(96, feat_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feat_dim), nn.ReLU(inplace=True), # 6x6
            nn.AdaptiveAvgPool2d(1),
        )
        self.feat_dim = feat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).flatten(1)


class TCNBlock(nn.Module):
    """
    Residual dilated 1D conv block over the temporal axis.
    Two conv layers at the same dilation, 'same' padding (centered, so the
    block is non-causal — see module docstring).
    """

    def __init__(self, channels: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        pad = dilation                      # (kernel 3 - 1) // 2 * dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=pad, dilation=dilation,
                      bias=False),
            nn.BatchNorm1d(channels), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 3, padding=pad, dilation=dilation,
                      bias=False),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.net(x))


class OnsetDetector(nn.Module):
    """
    (B, T, 2, H, W) -> (B, T) per-frame onset logits.

    Fully convolutional in time: train on fixed-length clips, then run a
    whole session in one pass at inference (no windowing, no stitching).
    """

    def __init__(self, feat_dim: int = FEAT_DIM,
                 dilations: tuple = TCN_DILATIONS, dropout: float = 0.1):
        super().__init__()
        self.encoder = FrameEncoder(feat_dim)
        self.tcn = nn.Sequential(*[
            TCNBlock(feat_dim, d, dropout) for d in dilations
        ])
        self.head = nn.Conv1d(feat_dim, 1, 1)

    @property
    def receptive_field(self) -> int:
        """Frames of temporal context feeding one output frame."""
        return 1 + sum(4 * d for d in
                       [b.net[0].dilation[0] for b in self.tcn])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t = x.shape[:2]
        feats = self.encoder(x.flatten(0, 1))        # (B*T, F)
        feats = feats.view(b, t, -1).transpose(1, 2) # (B, F, T)
        return self.head(self.tcn(feats)).squeeze(1) # (B, T)


def build_model(device: torch.device, dropout: float = 0.1) -> OnsetDetector:
    return OnsetDetector(dropout=dropout).to(device)


@torch.no_grad()
def score_stream(model: OnsetDetector, stream, device: torch.device,
                 chunk: int = 480) -> np.ndarray:
    """
    Per-frame onset probability for a whole session.

    The model is fully convolutional in time, so in principle a session
    goes through in one pass — but 1600 frames of encoder activations do
    not fit comfortably in memory, so it is chunked.

    Chunks are processed with `margin` extra frames of context on each side
    which are then DISCARDED. Without that, every chunk boundary would see
    zero-padded context instead of real frames and produce a band of ~60
    unreliable scores; with 4 chunks per session that would corrupt over
    10% of the output, right where peaks are being picked.
    """
    from dataset import to_tensor

    model.eval()
    n = len(stream)
    margin = model.receptive_field // 2
    out = np.empty(n, dtype=np.float32)

    start = 0
    while start < n:
        end = min(start + chunk, n)
        lo, hi = max(0, start - margin), min(n, end + margin)
        x = to_tensor(stream.clip_block(lo, hi - lo)).unsqueeze(0).to(device)
        logits = model(x)[0].float().cpu().numpy()
        out[start:end] = logits[start - lo:start - lo + (end - start)]
        start = end

    return 1.0 / (1.0 + np.exp(-out))
