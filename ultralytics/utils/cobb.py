# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""COBB utilities for continuous oriented bounding boxes."""

from __future__ import annotations

import math

import torch

from ultralytics.utils import ops


def _norm_angle(angle: torch.Tensor, angle_min: float = -math.pi / 4, angle_range: float = math.pi) -> torch.Tensor:
    """Normalize angles into a continuous range."""
    return torch.remainder(angle - angle_min, angle_range) + angle_min


def rotated_box_to_poly(rboxes: torch.Tensor) -> torch.Tensor:
    """Convert rotated boxes (xywhr) to polygons."""
    if rboxes.numel() == 0:
        return rboxes.new_zeros((0, 8))
    polys = ops.xywhr2xyxyxyxy(rboxes).reshape(-1, 8)
    return polys


def poly_to_rotated_box(polys: torch.Tensor) -> torch.Tensor:
    """Convert polygons (xyxyxyxy) to rotated boxes (xywhr)."""
    if polys.numel() == 0:
        return polys.new_zeros((0, 5))
    pts = polys.reshape(-1, 4, 2)
    pt1, pt2, pt3, pt4 = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]

    edge1 = torch.sqrt((pt1[:, 0] - pt2[:, 0]) ** 2 + (pt1[:, 1] - pt2[:, 1]) ** 2)
    edge2 = torch.sqrt((pt2[:, 0] - pt3[:, 0]) ** 2 + (pt2[:, 1] - pt3[:, 1]) ** 2)

    angles1 = torch.atan2(pt2[:, 1] - pt1[:, 1], pt2[:, 0] - pt1[:, 0])
    angles2 = torch.atan2(pt4[:, 1] - pt1[:, 1], pt4[:, 0] - pt1[:, 0])
    angles = torch.where(edge1 > edge2, angles1, angles2)
    angles = _norm_angle(angles)

    x_ctr = (pt1[:, 0] + pt3[:, 0]) / 2.0
    y_ctr = (pt1[:, 1] + pt3[:, 1]) / 2.0

    edges = torch.stack([edge1, edge2], dim=1)
    width = edges.max(1).values
    height = edges.min(1).values
    return torch.stack([x_ctr, y_ctr, width, height, angles], dim=1)


def rotated_box_to_bbox(rboxes: torch.Tensor) -> torch.Tensor:
    """Convert rotated boxes (xywhr) to horizontal boxes (xyxy)."""
    polys = rotated_box_to_poly(rboxes)
    xs = polys[:, 0::2]
    ys = polys[:, 1::2]
    return torch.stack([xs.min(1).values, ys.min(1).values, xs.max(1).values, ys.max(1).values], dim=1)


class COBBCoder:
    """Continuous OBB coder for ratio/score representation."""

    def __init__(self, pow_iou: float = 1.0, ratio_type: str = "sig") -> None:
        self.pow_iou = float(pow_iou)
        self.ratio_type = ratio_type

    @staticmethod
    def _safe_sqrt(value: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.clamp(value, min=0.0))

    @torch.no_grad()
    def build_iou_matrix(self, hbboxes: torch.Tensor, ratio_pred: torch.Tensor) -> torch.Tensor:
        """Build IoU matrix for the four COBB polygon types."""
        min_x, min_y, max_x, max_y = hbboxes.t()
        w = max_x - min_x
        h = max_y - min_y
        ratio_pred = ratio_pred.view(-1)
        w_large = w > h
        h_large = ~w_large
        eps = 1e-9

        x_ratio = torch.zeros_like(ratio_pred)
        y_ratio = torch.zeros_like(ratio_pred)

        if h_large.any():
            h_large_ratio = ratio_pred[h_large] / 4
            h_large_w = w[h_large]
            h_large_h = h[h_large].clamp(min=eps)
            h_large_delta_x = self._safe_sqrt(1 - 4 * h_large_ratio)
            x_ratio[h_large] = (1 - h_large_delta_x) / 2
            h_large_delta_y = self._safe_sqrt(
                1 - 4 * (h_large_w * h_large_w / (h_large_h * h_large_h)) * h_large_ratio
            )
            y_ratio[h_large] = (1 - h_large_delta_y) / 2

        if w_large.any():
            w_large_ratio = ratio_pred[w_large] / 4
            w_large_w = w[w_large].clamp(min=eps)
            w_large_h = h[w_large]
            w_large_delta_y = self._safe_sqrt(1 - 4 * w_large_ratio)
            y_ratio[w_large] = (1 - w_large_delta_y) / 2
            w_large_delta_x = self._safe_sqrt(
                1 - 4 * (w_large_h * w_large_h / (w_large_w * w_large_w)) * w_large_ratio
            )
            x_ratio[w_large] = (1 - w_large_delta_x) / 2

        l_01 = torch.sqrt((x_ratio * w) ** 2 + (y_ratio * h) ** 2)
        l_02 = torch.sqrt(((1 - x_ratio) * w) ** 2 + ((1 - y_ratio) * h) ** 2)
        l_03 = torch.sqrt((x_ratio * w) ** 2 + ((1 - y_ratio) * h) ** 2)
        l_04 = torch.sqrt(((1 - x_ratio) * w) ** 2 + (y_ratio * h) ** 2)

        one_minus_x = (1 - x_ratio).clamp(min=eps)
        one_minus_y = (1 - y_ratio).clamp(min=eps)
        i_01 = (1 - (1 - 2 * x_ratio) * x_ratio * w * w / (one_minus_y * h * h + eps)) * l_01 * l_02
        iou_01 = i_01 / (l_01 * l_02 + l_03 * l_04 - i_01 + eps)
        i_02 = (1 - (1 - 2 * y_ratio) * y_ratio * h * h / (one_minus_x * w * w + eps)) * l_01 * l_02
        iou_02 = i_02 / (l_01 * l_02 + l_03 * l_04 - i_02 + eps)

        i_03 = ((x_ratio + y_ratio - 2 * x_ratio * y_ratio) ** 2) / (one_minus_x * one_minus_y + eps) * w * h / 2
        iou_03 = torch.zeros_like(iou_02)
        valid = l_01 > 1e-5
        iou_03[valid] = i_03[valid] / (l_01[valid] * l_02[valid] * 2 - i_03[valid] + eps)

        h1 = 0.5 * w - (0.5 - y_ratio) / one_minus_y * w * x_ratio
        h2 = 0.5 * h - (0.5 - x_ratio) / one_minus_x * h * y_ratio
        s2 = h1 * h1 + h2 * h2
        tana = (0.5 - x_ratio) / one_minus_x * l_04 / (0.5 / one_minus_y * l_03 + eps)
        tanb = (0.5 - y_ratio) / one_minus_y * l_03 / (0.5 / one_minus_x * l_04 + eps)
        nzero = tana + tanb > 1e-8
        i_12 = torch.zeros_like(iou_02)
        i_12[nzero] = (tana[nzero] * tanb[nzero] / (tana[nzero] + tanb[nzero])) * s2[nzero] * 2 + h1[nzero] * h2[
            nzero
        ] * 2
        iou_12 = i_12 / (l_03 * l_04 * 2 - i_12 + eps)

        iou_self = torch.ones_like(ratio_pred)
        iou0 = torch.stack([iou_self, iou_01, iou_02, iou_03], dim=-1)
        iou1 = torch.stack([iou_01, iou_self, iou_12, iou_02], dim=-1)
        iou2 = torch.stack([iou_02, iou_12, iou_self, iou_01], dim=-1)
        iou3 = torch.stack([iou_03, iou_02, iou_01, iou_self], dim=-1)
        return torch.stack([iou0, iou1, iou2, iou3], dim=-2)

    def build_polypairs(self, hbboxes: torch.Tensor, ratio_pred: torch.Tensor) -> list[torch.Tensor]:
        """Build four polygon candidates for each horizontal box and ratio."""
        min_x, min_y, max_x, max_y = hbboxes.t()
        w = max_x - min_x
        h = max_y - min_y
        ratio_pred = ratio_pred.view(-1)
        w_large = w > h
        h_large = ~w_large
        eps = 1e-9

        x1 = torch.zeros_like(ratio_pred)
        x2 = torch.zeros_like(ratio_pred)
        y1 = torch.zeros_like(ratio_pred)
        y2 = torch.zeros_like(ratio_pred)

        if h_large.any():
            h_large_ratio = ratio_pred[h_large] / 4
            h_large_w = w[h_large]
            h_large_h = h[h_large].clamp(min=eps)
            h_large_delta_x = self._safe_sqrt(1 - 4 * h_large_ratio)
            x1[h_large] = (1 - h_large_delta_x) / 2 * h_large_w
            x2[h_large] = (1 + h_large_delta_x) / 2 * h_large_w
            h_large_delta_y = self._safe_sqrt(
                1 - 4 * (h_large_w * h_large_w / (h_large_h * h_large_h)) * h_large_ratio
            )
            y1[h_large] = (1 - h_large_delta_y) / 2 * h_large_h
            y2[h_large] = (1 + h_large_delta_y) / 2 * h_large_h

        if w_large.any():
            w_large_ratio = ratio_pred[w_large] / 4
            w_large_w = w[w_large].clamp(min=eps)
            w_large_h = h[w_large]
            w_large_delta_y = self._safe_sqrt(1 - 4 * w_large_ratio)
            y1[w_large] = (1 - w_large_delta_y) / 2 * w_large_h
            y2[w_large] = (1 + w_large_delta_y) / 2 * w_large_h
            w_large_delta_x = self._safe_sqrt(
                1 - 4 * (w_large_h * w_large_h / (w_large_w * w_large_w)) * w_large_ratio
            )
            x1[w_large] = (1 - w_large_delta_x) / 2 * w_large_w
            x2[w_large] = (1 + w_large_delta_x) / 2 * w_large_w

        poly1 = torch.stack(
            [min_x + x1, min_y, max_x, min_y + y2, max_x - x1, max_y, min_x, max_y - y2], dim=-1
        )
        poly2 = torch.stack(
            [min_x + x2, min_y, max_x, min_y + y2, max_x - x2, max_y, min_x, max_y - y2], dim=-1
        )
        poly3 = torch.stack(
            [min_x + x1, min_y, max_x, min_y + y1, max_x - x1, max_y, min_x, max_y - y1], dim=-1
        )
        poly4 = torch.stack(
            [min_x + x2, min_y, max_x, min_y + y1, max_x - x2, max_y, min_x, max_y - y1], dim=-1
        )
        return [
            poly_to_rotated_box(poly1),
            poly_to_rotated_box(poly2),
            poly_to_rotated_box(poly3),
            poly_to_rotated_box(poly4),
        ]

    @torch.no_grad()
    def encode(self, rbboxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode rotated boxes into ratio and score targets."""
        polys = rotated_box_to_poly(rbboxes)
        xs = polys[:, 0::2]
        ys = polys[:, 1::2]
        hbboxes = torch.stack([xs.min(1).values, ys.min(1).values, xs.max(1).values, ys.max(1).values], dim=1)

        polys = polys.view(-1, 4, 2)
        w = hbboxes[:, 2] - hbboxes[:, 0]
        h = hbboxes[:, 3] - hbboxes[:, 1]
        x_ind = torch.argsort(polys[:, :, 0], dim=1)
        y_ind = torch.argsort(polys[:, :, 1], dim=1)
        polys_x = polys[:, :, 0]
        polys_y = polys[:, :, 1]
        index = torch.arange(polys.shape[0], device=polys.device)
        s_x = polys_x[index, x_ind[:, 1]]
        s_y = polys_y[index, y_ind[:, 1]]
        dx = (s_x - hbboxes[:, 0]) / w
        dy = (s_y - hbboxes[:, 1]) / h

        w_large = w > h
        ratio = torch.zeros_like(dx)
        ratio[~w_large] = dx[~w_large] * (1 - dx[~w_large]) * 4
        ratio[w_large] = dy[w_large] * (1 - dy[w_large]) * 4
        ratio = ratio.clamp(max=1.0)

        ious = self.build_iou_matrix(hbboxes, ratio)
        is_type13 = (x_ind[:, 1] == y_ind[:, 2]) | (x_ind[:, 1] == y_ind[:, 3])
        is_type23 = (x_ind[:, 0] == y_ind[:, 2]) | (x_ind[:, 0] == y_ind[:, 3])
        rtype = is_type23.long() * 2 + is_type13.long()
        ious = ious[torch.arange(ious.shape[0], device=ious.device), rtype]
        ious = ious.pow(self.pow_iou)

        if self.ratio_type == "sig":
            ratio = 1 - torch.sqrt(1 - ratio)
        elif self.ratio_type == "ln":
            ratio = (1 - torch.sqrt(1 - ratio)) / 2
            square_like = (rtype == 1) | (rtype == 2)
            ratio[square_like] = 1 - ratio[square_like]
            ratio = 1 + torch.log2(ratio.clamp(min=1e-6))
        else:
            raise NotImplementedError(f"Unsupported ratio_type: {self.ratio_type}")

        return ratio[:, None], ious

    def decode(self, hbboxes: torch.Tensor, ratio_pred: torch.Tensor, rotated_scores: torch.Tensor) -> torch.Tensor:
        """Decode horizontal boxes with ratio and score predictions into rotated boxes."""
        if hbboxes.numel() == 0:
            return hbboxes.new_zeros((0, 5))

        ratio_pred = ratio_pred.view(-1)
        if self.ratio_type == "sig":
            ratio_pred = ratio_pred.clamp(0, 1)
            ratio_pred = 1 - (1 - ratio_pred) ** 2
        elif self.ratio_type == "ln":
            square_like = ratio_pred > 0
            ratio_pred = torch.clamp(torch.pow(2.0, ratio_pred - 1), min=0.0, max=1.0)
            ratio_pred[square_like] = 1 - ratio_pred[square_like]
            ratio_pred = 1 - (1 - ratio_pred * 2) ** 2
        else:
            raise NotImplementedError(f"Unsupported ratio_type: {self.ratio_type}")

        if rotated_scores.dim() == 3:
            rotated_scores = rotated_scores.squeeze(-1)
        rbboxes = torch.stack(self.build_polypairs(hbboxes, ratio_pred), dim=1)
        best_index = rotated_scores.argmax(dim=-1)
        return rbboxes[torch.arange(rbboxes.shape[0], device=rbboxes.device), best_index]
