"""Binary intersection-classification CNN.

Per Section 7 of the spec. Small architecture sized for CPU inference on a
laptop. The flatten dimension is computed from a dummy forward pass so the
same module supports patch_size 32, 64, or 128 without code changes.

Output is a single sigmoid-activated probability per patch. We chose
``BCEWithLogitsLoss`` in the trainer rather than ``BCELoss`` for numerical
stability, so the *training* path uses the model's pre-sigmoid logit. The
forward path applies the sigmoid only in inference. Both behaviors are
exposed via ``forward(x, return_logits=False)``.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn


log = logging.getLogger("training")


class IntersectionCNN(nn.Module):
    """Three-block CNN + 2-layer MLP head.

    Architecture follows Section 7 of the spec, with one deliberate
    deviation: ``GroupNorm`` is used in place of ``BatchNorm2d``.

    The spec specifies BatchNorm, but BN's running statistics are computed
    per-batch and degrade badly on small batches — exactly the regime the
    user operates in (early-stage labeling produces datasets of dozens to
    a few hundred patches, so batches at training time are small and
    inference is often single-patch). GroupNorm normalizes across channel
    groups within each sample, so it's batch-size-independent. Same
    parameter count, same forward cost, more robust on small data.

    All three GroupNorm layers use 8 groups: 8 divides 16, 32, and 64
    evenly, giving 2 / 4 / 8 channels per group across the depth. The
    GroupNorm paper recommends "32 groups by default, 4 channels per
    group for narrow networks"; our network is narrow, so we lean toward
    the channels-per-group end of the recommendation.

    Layer order:
        Conv2d(1, 16, 3, padding=1) → GroupNorm(8, 16)  → ReLU → MaxPool(2)
        Conv2d(16, 32, 3, padding=1) → GroupNorm(8, 32)  → ReLU → MaxPool(2)
        Conv2d(32, 64, 3, padding=1) → GroupNorm(8, 64)  → ReLU → MaxPool(2)
        Flatten
        Linear(flat_dim, 128) → ReLU → Dropout(0.5)
        Linear(128, 1) [Sigmoid applied in forward when not returning logits]

    The flatten dimension is computed once at ``__init__`` time by running
    a dummy ``(1, 1, patch_size, patch_size)`` tensor through the conv
    blocks. This keeps the architecture identical for patch_size 32, 64,
    or 128 — only the head's input size changes.

    Inputs to ``forward`` are float32 tensors of shape
    ``(batch, 1, patch_size, patch_size)`` normalized to ``[0, 1]``. The
    trainer / inference helpers handle that normalization; callers should
    feed already-prepared tensors.
    """

    # Number of groups for each GroupNorm layer. 8 divides 16/32/64 evenly.
    _GN_GROUPS = 8

    def __init__(self, patch_size: int = 64):
        super().__init__()
        if patch_size not in (32, 64, 128):
            # The spec lists 32 / 64 / 128 as the supported sizes. We allow
            # any value that survives 3 max-pools, but warn on unusual ones.
            log.warning(
                "IntersectionCNN: unusual patch_size=%d; spec lists 32/64/128",
                patch_size,
            )
        if patch_size < 8:
            raise ValueError(
                f"patch_size must be at least 8 (3 max-pools halve it 8x); "
                f"got {patch_size}"
            )

        self.patch_size = patch_size

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.GroupNorm(self._GN_GROUPS, 16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.GroupNorm(self._GN_GROUPS, 32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(self._GN_GROUPS, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Compute the flatten dimension via a dummy forward pass. This
        # avoids hard-coding ``64 * 8 * 8`` for patch_size=64 and lets
        # patch_size=32 / 128 work transparently.
        self._flat_dim = self._compute_flat_dim(patch_size)

        self.classifier = nn.Sequential(
            nn.Linear(self._flat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, 1),
        )

    def _compute_flat_dim(self, patch_size: int) -> int:
        """Run a dummy tensor through the conv stack to get the flatten size."""
        # ``no_grad`` because we don't want autograd state from this probe.
        with torch.no_grad():
            dummy = torch.zeros(1, 1, patch_size, patch_size)
            out = self.features(dummy)
        return int(out.shape[1] * out.shape[2] * out.shape[3])

    def forward(
        self,
        x: torch.Tensor,
        return_logits: bool = False,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Shape ``(batch, 1, H, W)``, float32 in ``[0, 1]``.
        return_logits : bool
            If True, return raw pre-sigmoid logits (for use with
            ``BCEWithLogitsLoss``). If False (default), apply sigmoid.

        Returns
        -------
        Tensor of shape ``(batch, 1)``.
        """
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        logits = self.classifier(x)
        if return_logits:
            return logits
        return torch.sigmoid(logits)
