"""Custom IoU loss for bounding box regression.

Boxes are expected in (x_center, y_center, width, height) format in pixel space.
Loss = 1 - IoU, which lies in [0, 1]:
  - 0 = perfect overlap (ideal)
  - 1 = no overlap at all

For localization training, use: total_loss = MSELoss(pred, target) + IoULoss(pred, target)
The MSE loss handles coordinate regression while IoU loss directly optimises the
overlap metric that matters for detection quality.
"""

import torch
import torch.nn as nn


class IoULoss(nn.Module):
    """IoU loss for bounding box regression.

    Converts (cx, cy, w, h) → (x1, y1, x2, y2) internally, computes IoU,
    then returns 1 - IoU as the loss (so lower = better, minimum = 0).

    Numerical stability: eps prevents division by zero when union_area = 0
    (which happens when both pred and target have zero area).
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        """
        Args:
            eps:       Small constant to avoid division by zero.
            reduction: 'mean' | 'sum' | 'none'.
                       'mean' (default) — average loss over the batch.
                       'sum'  — sum of per-sample losses.
                       'none' — return per-sample loss tensor [B].
        """
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(
                f"reduction must be 'none', 'mean', or 'sum', got '{reduction}'"
            )
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute IoU loss.

        Args:
            pred_boxes:   [B, 4] predicted boxes   in (cx, cy, w, h) pixel format.
            target_boxes: [B, 4] ground-truth boxes in (cx, cy, w, h) pixel format.

        Returns:
            Scalar loss (or [B] tensor if reduction='none').
        """
        # ── Convert (cx, cy, w, h) → (x1, y1, x2, y2) ──────────────────────
        pred_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
        pred_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
        pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
        pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2

        tgt_x1 = target_boxes[:, 0] - target_boxes[:, 2] / 2
        tgt_y1 = target_boxes[:, 1] - target_boxes[:, 3] / 2
        tgt_x2 = target_boxes[:, 0] + target_boxes[:, 2] / 2
        tgt_y2 = target_boxes[:, 1] + target_boxes[:, 3] / 2

        # ── Intersection ─────────────────────────────────────────────────────
        inter_x1 = torch.max(pred_x1, tgt_x1)
        inter_y1 = torch.max(pred_y1, tgt_y1)
        inter_x2 = torch.min(pred_x2, tgt_x2)
        inter_y2 = torch.min(pred_y2, tgt_y2)

        # clamp(min=0): when boxes don't overlap, inter dims are negative → clamp to 0
        inter_w    = (inter_x2 - inter_x1).clamp(min=0)
        inter_h    = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h                      # [B]

        # ── Union ─────────────────────────────────────────────────────────────
        pred_area = (pred_x2 - pred_x1).clamp(min=0) * (pred_y2 - pred_y1).clamp(min=0)
        tgt_area  = (tgt_x2  - tgt_x1).clamp(min=0) * (tgt_y2  - tgt_y1).clamp(min=0)
        union_area = pred_area + tgt_area - inter_area      # [B]

        # ── IoU and loss ──────────────────────────────────────────────────────
        iou  = inter_area / (union_area + self.eps)         # [B], in [0, 1]
        loss = 1.0 - iou                                    # [B], in [0, 1]

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # "none"
            return loss


# """Custom IoU loss 
# """

# import torch
# import torch.nn as nn

# class IoULoss(nn.Module):
#     """IoU loss for bounding box regression.
#     """

#     def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
#         """
#         Initialize the IoULoss module.
#         Args:
#             eps: Small value to avoid division by zero.
#             reduction: Specifies the reduction to apply to the output: 'mean' | 'sum'.
#         """
#         super().__init__()
#         self.eps = eps
#         self.reduction = reduction
#         # TODO: validate reduction in {"none", "mean", "sum"}.

#     def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
#         """Compute IoU loss between predicted and target bounding boxes.
#         Args:
#             pred_boxes: [B, 4] predicted boxes in (x_center, y_center, width, height) format.
#             target_boxes: [B, 4] target boxes in (x_center, y_center, width, height) format."""
#         # TODO: implement IoU loss.
#         raise NotImplementedError("Implement IoULoss.forward")