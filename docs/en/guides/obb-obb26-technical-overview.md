---
comments: true
description: A systematic technical overview of rotated object detection labels, OBB and OBB26 principles, angle specifications, loss functions, datasets, and evaluation metrics for academic reference.
keywords: OBB, OBB26, oriented bounding box, rotated object detection, HBB, YOLO, angle regression, probiou, DOTA, object detection, bounding box
---

# Rotated Object Detection Labels, OBB and OBB26: Technical Overview

## 1. Introduction

Rotated object detection holds significant importance across a wide range of general computer vision applications. Conventional object detection methods typically use **Horizontal Bounding Boxes (HBB)** for object localization. However, when objects are tilted, rotated, or distributed at arbitrary orientations, HBBs cannot tightly fit object boundaries, introducing background noise and causing severe box overlap in densely packed scenarios, thereby degrading detection accuracy.

To address this, **Oriented Bounding Boxes (OBB)** introduce an additional rotation angle parameter, enabling a rotated rectangle to precisely enclose targets in any direction. OBB-based methods have demonstrated substantial improvements in aerial imagery, vehicle detection, document analysis, and industrial inspection.

This document provides a systematic review grounded in the actual Ultralytics YOLO project implementation, covering: rotated label definitions, OBB vs HBB comparisons, OBB and OBB26 principles, angle specification differences, loss function analysis, OBB26 improvements relative to prior work, and representative datasets and evaluation metrics.

---

## 2. Rotated Image Label Definition Formats

### 2.1 Horizontal Bounding Box (HBB)

The horizontal bounding box is the most common annotation format in object detection:

$$
\text{HBB} = (c_x, c_y, w, h)
$$

where $c_x, c_y$ are the center coordinates and $w, h$ are width and height. HBBs are always axis-aligned (AABB), providing no orientation information.

**Key limitations of HBB:**

- Tilted objects lead to large axis-aligned boxes containing excessive background;
- Densely packed rotated objects (e.g., vehicles in parking lots, ships in ports) produce heavily overlapping boxes, degrading NMS performance;
- Orientation information is entirely absent, limiting downstream direction-aware tasks.

### 2.2 Oriented Bounding Box (OBB)

OBBs extend HBB with a rotation angle parameter:

$$
\text{OBB} = (c_x, c_y, w, h, \theta)
$$

where $\theta$ is the rotation angle relative to a reference direction. OBBs tightly fit arbitrarily oriented rectangular targets, minimizing enclosing area and improving detection accuracy and NMS quality.

**OBB vs HBB Comparison:**

| Property | HBB | OBB |
|----------|-----|-----|
| Rotated target fitting | ❌ Large area, much background | ✅ Tight fit, minimal bounding box |
| Dense target overlap | ❌ IoU inflated, NMS over-suppresses | ✅ Accurate IoU, correct suppression |
| Orientation information | ❌ Not available | ✅ Directly encodes rotation angle |
| Annotation complexity | ✅ Simple (4 parameters) | ❌ More complex (5 params / 8 points) |
| Inference complexity | ✅ Standard NMS | ❌ Requires rotated NMS |

### 2.3 OBB Annotation Format

In the Ultralytics YOLO project, the **annotation format** (label files) for OBB uses **4 normalized corner points**:

```
class_index  x1 y1  x2 y2  x3 y3  x4 y4
```

where $(x_i, y_i)$ are the normalized coordinates of the four vertices (range $[0, 1]$), listed in order. Example:

```
0  0.780811 0.743961  0.782371 0.74686  0.777691 0.752174  0.776131 0.749758
```

**Internal processing format (`xywhr`)**: During loss computation and prediction output, YOLO converts the 4-corner format to `xywhr`:

$$
\text{xywhr} = (c_x, c_y, w, h, r)
$$

where $r$ is the rotation angle in radians. This format is convenient for parametric regression and consistent with rotated IoU computation.

### 2.4 OBB26 Label Representation

OBB26 is the detection head variant used for rotated object detection in the Ultralytics YOLO26 series. From a label format perspective, OBB26 uses the **same annotation format** as OBB (`class_index x1 y1 x2 y2 x3 y3 x4 y4`). The key differences lie in the **network angle prediction processing**, detailed in Section 4.

---

## 3. OBB Angle Specification Differences and YOLO Integration

### 3.1 Angle Definition Convention Diversity

The rotation angle $\theta$ in oriented bounding boxes has multiple definition conventions, which can vary significantly across datasets, toolkits, and research works.

#### 3.1.1 OpenCV Convention

`cv2.minAreaRect` returns $\theta \in [-90°, 0°)$:
- Defined as the **clockwise** angle from the positive x-axis to the **"width" edge** of the returned rectangle; the "width" edge is whichever side OpenCV associates with that angle and is **not guaranteed to be the longer side**;
- Clockwise rotation is negative; a horizontal box returns $\theta = 0°$; rotating to vertical approaches $\theta = -90°$.

#### 3.1.2 Short-Edge Counter-Clockwise Convention

Some datasets (including DOTA) and research papers adopt a **short-edge counter-clockwise** convention:
- $\theta \in [0°, 90°)$, defined as the counter-clockwise angle between the short edge and the horizontal axis;
- Ambiguous when width equals height (requires additional convention).

#### 3.1.3 YOLO Sigmoid-Mapped Convention (Original OBB Head)

In Ultralytics YOLO's OBB head, the angle is in radians, constrained via Sigmoid activation:

$$
\theta = (\text{sigmoid}(x) - 0.25) \times \pi \in \left[-\frac{\pi}{4}, \frac{3\pi}{4}\right]
$$

This maps the angle to $[-45°, 135°)$. The $-0.25\pi$ bias shifts the typical $[0°, \pi)$ range leftward to better cover common rotation angle distributions.

!!! note "YOLO OBB Angle Constraint"
    Per the project documentation, OBB angles are constrained to the range **0–90 degrees (exclusive of 90)**. Angles ≥ 90° are not directly supported and should be avoided during annotation.

#### 3.1.4 OBB26 Unconstrained Convention

The OBB26 head outputs **raw, unconstrained angle predictions** without Sigmoid activation. Angle periodicity is handled explicitly by the loss function (see Section 4.2).

#### 3.1.5 Comparison of Conventions

| Convention | Range | Periodicity Handling | Gradient at Boundary |
|-----------|-------|---------------------|---------------------|
| OpenCV | $[-90°, 0°)$ | Boundary jump | Discontinuous |
| Short-edge CCW | $[0°, 90°)$ | Boundary jump | Discontinuous |
| YOLO Sigmoid | $[-45°, 135°)$ | Sigmoid suppression | Vanishing near limits |
| OBB26 Unconstrained | Raw prediction | Loss-level wrapping | Smooth everywhere |

**Core challenge**: The $180°$ periodicity of rectangles (angle $\theta$ and $\theta + 180°$ produce identical boxes) creates "multi-solution" ambiguity in regression targets. This inconsistency during gradient backpropagation leads to training instability—a fundamental problem that motivates most angle-aware loss design.

#### 3.1.6 Why the YOLO xywhr Angle Is [-45°, 135°) Instead of [-90°, 0°)

The YOLO internal representation (`xywhr`) is produced by `xyxyxyxy2xywhr` (`ultralytics/utils/ops.py`), which calls `cv2.minAreaRect` and then applies two transformations:

**Step 1 — OpenCV baseline**

```python
(cx, cy), (w, h), angle = cv2.minAreaRect(pts)
theta = angle / 180 * np.pi          # degrees → radians; theta ∈ [-π/2, 0)
```

OpenCV returns `angle ∈ [-90°, 0°)`. The "width" `w` in the result can be either the longer **or** shorter dimension — OpenCV imposes no constraint on which is larger.

**Step 2 — Enforce w ≥ h (long side is always called w)**

```python
if w < h:
    w, h = h, w
    theta += np.pi / 2   # the new long axis is 90° away from the old one
```

When OpenCV's `w < h`, the "width" edge is actually the shorter side. YOLO swaps the labels so that `w` always denotes the longer dimension. Geometrically, the longer edge's angle is exactly 90° greater than the shorter edge's angle, so `theta += π/2` corrects the direction reference.

After the conditional swap:

| Case | OpenCV output | theta after swap |
|------|--------------|-----------------|
| `w ≥ h` already | `w ≥ h`, `angle ∈ [-90°, 0°)` | `theta ∈ [-π/2, 0)` |
| `w < h` (swap) | `w < h`, `angle ∈ [-90°, 0°)` | `theta ∈ [0, π/2)` |

**Step 3 — Normalize to [-π/4, 3π/4)**

```python
while theta >= 3 * np.pi / 4:
    theta -= np.pi          # wrap down from above 135°
while theta < -np.pi / 4:
    theta += np.pi          # wrap up from below -45°
```

Applying these loops to each range from Step 2:

| theta before loops | Result after loops |
|-------------------|-------------------|
| `[-π/2, -π/4)` (no-swap, steep angle) | `+π` → `[π/2, 3π/4)` i.e. `[90°, 135°)` |
| `[-π/4, 0)` (no-swap, shallow angle) | unchanged → `[-π/4, 0)` i.e. `[-45°, 0°)` |
| `[0, π/2)` (after swap) | unchanged → `[0, π/2)` i.e. `[0°, 90°)` |

Combined final range: $[-\pi/4,\ 0) \cup [0,\ \pi/2) \cup [\pi/2,\ 3\pi/4) = [-\pi/4,\ 3\pi/4)$, i.e. **[-45°, 135°)**.

**Summary of the key difference**

OpenCV's angle describes the orientation of whichever edge it internally labels as the "width" (may be the short side). YOLO's angle always describes the orientation of the **long side** (`w ≥ h`). This semantic redefinition—together with the compensating +90° shift and the normalization loops—transforms OpenCV's `[-90°, 0°)` into YOLO's `[-45°, 135°)`.

The sigmoid formula in the OBB detection head, `(sigmoid(x) − 0.25) × π`, was designed to exactly match this output range:

$$
\sigma(x) \in (0,\ 1) \;\Longrightarrow\; (\sigma(x) - 0.25)\pi \in (-\tfrac{\pi}{4},\ \tfrac{3\pi}{4})
$$

### 3.2 OBB Implementation in YOLO

#### 3.2.1 Label Conversion: 4-Corner Points → xywhr (`xyxyxyxy2xywhr`)

The full conversion from annotation 4-corner format to the internal `xywhr` representation is in `ultralytics/utils/ops.py`:

```python
def xyxyxyxy2xywhr(x):
    """Convert [xy1,xy2,xy3,xy4] (N,8) → [cx,cy,w,h,theta] (N,5), theta in [-pi/4, 3pi/4)."""
    is_torch = isinstance(x, torch.Tensor)
    points = x.cpu().numpy() if is_torch else x
    points = points.reshape(len(x), -1, 2)
    rboxes = []
    for pts in points:
        # Step 1: call OpenCV — returns angle in [-90°, 0°), w may be < h
        (cx, cy), (w, h), angle = cv2.minAreaRect(pts)

        # Step 2: convert degrees → radians
        theta = angle / 180 * np.pi                      # theta ∈ [-π/2, 0)

        # Step 3: enforce w >= h (long axis is always w)
        if w < h:
            w, h = h, w
            theta += np.pi / 2                           # long axis is 90° away

        # Step 4: normalize to [-π/4, 3π/4)
        while theta >= 3 * np.pi / 4:
            theta -= np.pi
        while theta < -np.pi / 4:
            theta += np.pi

        rboxes.append([cx, cy, w, h, theta])
    return torch.tensor(rboxes, ...) if is_torch else np.asarray(rboxes)
```

This is the direct reason why YOLO's xywhr angle is in **[-45°, 135°)** rather than OpenCV's [-90°, 0°) — see Section 3.1.6 for a detailed walkthrough.

#### 3.2.2 Detection Head Architecture

In `ultralytics/nn/modules/head.py`, the `OBB` class extends `Detect` with a dedicated angle prediction branch:

```python
class OBB(Detect):
    def __init__(self, nc=80, ne=1, reg_max=16, end2end=False, ch=()):
        super().__init__(nc, reg_max, end2end, ch)
        self.ne = ne  # number of angle parameters (default: 1)
        c4 = max(ch[0] // 4, self.ne)
        # Dedicated angle prediction convolution branch
        self.cv4 = nn.ModuleList(
            nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.ne, 1))
            for x in ch
        )
```

Angle prediction with Sigmoid activation (original OBB):

```python
def forward_head(self, x, box_head, cls_head, angle_head):
    # ...
    angle = (angle.sigmoid() - 0.25) * math.pi  # mapped to [-π/4, 3π/4]
    preds["angle"] = angle
```

#### 3.2.3 Rotated Box Decoding (`dist2rbox`)

YOLO uses the **Distribution Focal Loss (DFL)** framework, predicting box parameters as distance distributions from anchor points (ltrb). For rotated boxes, the `dist2rbox` function handles decoding:

```python
def dist2rbox(pred_dist, pred_angle, anchor_points, dim=-1):
    lt, rb = pred_dist.split(2, dim=dim)
    cos, sin = torch.cos(pred_angle), torch.sin(pred_angle)
    xf, yf = ((rb - lt) / 2).split(1, dim=dim)
    x, y = xf * cos - yf * sin, xf * sin + yf * cos
    xy = torch.cat([x, y], dim=dim) + anchor_points
    return torch.cat([xy, lt + rb], dim=dim)
```

This decodes predicted ltrb distances and angle into rotated box center and dimensions in `xywh` format.

#### 3.2.4 NMS Post-processing

Rotated NMS uses **Probabilistic IoU (probiou)** for efficient rotated box IoU computation, avoiding expensive polygon intersection calculations:

```python
# Rotated NMS uses batch_probiou for IoU
i = TorchNMS.fast_nms(boxes, scores, iou_thres, iou_func=batch_probiou)
```

---

## 4. OBB26: Principles and Improvements

### 4.1 Core Differences Between OBB and OBB26

The `OBB26` class (`ultralytics/nn/modules/head.py`) extends `OBB` with a key improvement: **removing the Sigmoid activation on angle prediction**:

```python
class OBB26(OBB):
    def forward_head(self, x, box_head, cls_head, angle_head):
        preds = Detect.forward_head(self, x, box_head, cls_head)
        if angle_head is not None:
            bs = x[0].shape[0]
            angle = torch.cat(
                [angle_head[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2
            )  # Raw angle prediction — no Sigmoid constraint
            preds["angle"] = angle
        return preds
```

| Property | OBB (original) | OBB26 (improved) |
|----------|---------------|-----------------|
| Angle activation | Sigmoid → $[-\pi/4, 3\pi/4]$ | None (unconstrained) |
| Angle range | Fixed $[-45°, 135°)$ | Freely learned |
| Periodicity handling | Sigmoid suppresses boundary | Loss-level modular wrapping |
| Architecture | YOLO early OBB | YOLO26 series |

**Design motivation**: While the Sigmoid constraint limits angle regression range, near the boundaries ($-45°$ or $135°$) the Sigmoid gradient approaches zero, causing vanishing gradients and training instability. OBB26 removes this constraint, relying on a periodicity-aware loss function for stable angle regression.

### 4.2 Periodicity-Aware Angle Loss (`calculate_angle_loss`)

OBB26 uses a **period-aware sin-square angle loss** with aspect-ratio-adaptive weighting:

```python
def calculate_angle_loss(self, pred_bboxes, target_bboxes, fg_mask, weight,
                          target_scores_sum, lambda_val=3):
    w_gt = target_bboxes[..., 2]
    h_gt = target_bboxes[..., 3]
    pred_theta = pred_bboxes[..., 4]
    target_theta = target_bboxes[..., 4]

    # Aspect ratio weight: near-square boxes are insensitive to angle, low weight
    log_ar = torch.log((w_gt + 1e-9) / (h_gt + 1e-9))
    scale_weight = torch.exp(-(log_ar**2) / (lambda_val**2))

    # Modular angle difference wrapping (π period)
    delta_theta = pred_theta - target_theta
    delta_theta_wrapped = delta_theta - torch.round(delta_theta / math.pi) * math.pi

    # Sin-square angle loss
    ang_loss = torch.sin(2 * delta_theta_wrapped[fg_mask]) ** 2
    ang_loss = scale_weight[fg_mask] * ang_loss * weight
    return ang_loss.sum() / target_scores_sum
```

**Key design elements:**

1. **Modular wrapping ($\pi$ mod)**: `delta_theta_wrapped = delta_theta - round(delta_theta/π) * π` maps the angle difference to $[-\pi/2, \pi/2)$, eliminating gradient direction inconsistency caused by the $180°$ symmetry of rectangles.

2. **Sin-square loss**: `sin(2Δθ)²` is a smooth periodic function with a zero at $\Delta\theta = 0$ and maximum at $\pm\pi/4$. It is second-order smooth near zero, avoiding the non-differentiability of L1 and the high variance of L2 in angular regression.

3. **Aspect-ratio adaptive weight**: `scale_weight = exp(-(log(w/h))²/λ²)` reduces the angle loss contribution for near-square targets (where angle ambiguity has little impact on detection quality) and approaches 1.0 for elongated targets where angle accuracy is critical.

**Formulation summary:**

$$
\mathcal{L}_{angle} = \frac{1}{N_+} \sum_{i \in +} w_i \cdot \sin^2(2\Delta\theta_{\text{wrapped},i})
$$

$$
\Delta\theta_{\text{wrapped}} = \Delta\theta - \text{round}\!\left(\frac{\Delta\theta}{\pi}\right) \cdot \pi, \qquad
w_i = \exp\!\left(-\frac{(\ln(w_{gt}/h_{gt}))^2}{\lambda^2}\right)
$$

### 4.3 YOLO26-OBB Architecture

In the YOLO26-obb model configuration (`ultralytics/cfg/models/26/yolo26-obb.yaml`):

```yaml
# Head uses OBB26 detection head
- [[16, 19, 22], 1, OBB26, [nc, 1]]  # OBB26(P3, P4, P5)
```

The model uses `end2end: True` with lightweight `reg_max: 1` DFL, reducing multi-bin DFL overhead and enabling NMS-free end-to-end training (one-to-one assignment).

| Scale | Parameters | GFLOPs |
|-------|-----------|--------|
| nano (n) | 2.7M | 16.9 |
| small (s) | 10.6M | 63.5 |
| medium (m) | 23.6M | 211.9 |
| large (l) | 28.0M | 259.0 |
| extra-large (x) | 62.8M | 578.9 |

---

## 5. Comparison with Published Methods: Improvements and Trade-offs

### 5.1 Evolution of Rotated Object Detection

#### 5.1.1 Anchor-Based Early Methods

**Representative works**: R2CNN, RRPN, RoI Transformer

- **Strengths**: Leverage mature Faster R-CNN framework; high accuracy;
- **Weaknesses**: Direct angle regression requires additional constraints; severe **boundary discontinuity** (numerical jump at $0°/90°$ boundaries destabilizes gradients); slow two-stage inference.

#### 5.1.2 Angle Encoding Improvements

**CSL (Circular Smooth Label, ECCV 2020)**: Converts angle regression into classification with 180 soft-labeled intervals using circular smoothing:
- Strengths: Completely eliminates boundary jumps; stable predictions;
- Weaknesses: 180-channel classification head increases parameter count; accuracy depends on interval resolution.

**DCL (CVPR 2021)**: Uses Gray coding to compress the CSL classification head:
- Strengths: More efficient than CSL;
- Weaknesses: Complex encoding/decoding; slight residual discontinuity at boundary.

**GWD (CVPR 2021)**: Models rotated boxes as 2D Gaussian distributions, using Wasserstein distance as loss:
- Strengths: Bypasses explicit angle regression entirely; robust to angular periodicity;
- Weaknesses: Gaussian approximation is imperfect; degrades for near-square targets.

**KFIoU (ICLR 2022)**: Kalman filter-based IoU approximation improving on GWD with more accurate overlap estimation:
- Strengths: Good scale consistency; strong DOTA results;
- Weaknesses: Complex implementation; introduces additional hyperparameters.

#### 5.1.3 Single-Stage YOLO-Based Methods

**S2ANet (TGRS 2021)**: Introduces Feature Alignment Module and axis-aligned NMS:
- Strengths: High accuracy, strong DOTA performance;
- Weaknesses: Complex architecture, slower inference.

**Ultralytics YOLO26-OBB / OBB26**: Integrates OBB26 detection head into lightweight single-stage YOLO, end-to-end training:
- Strengths: Fast, simple deployment, fully compatible with YOLO ecosystem;
- Weaknesses: Extreme-angle handling (near $\pm45°$ or $135°$ boundaries) relies on loss function rather than structural guarantees.

### 5.2 Structural Variations

#### 5.2.1 Angle Parameterization Methods

| Method | Parameterization | Characteristics |
|--------|-----------------|----------------|
| Direct regression | $\theta \in [0°, 90°)$ | Simple but severe boundary issues |
| Sigmoid mapping (OBB) | $\sigma(x) \times \pi$ | Bounded range, stable gradients away from boundary |
| Unconstrained raw (OBB26) | Raw logits + loss wrapping | Flexible, requires periodic loss |
| Sine/cosine decomposition | $(\cos 2\theta, \sin 2\theta)$ | Continuous, unambiguous; doubles angle parameters |
| Soft classification (CSL) | 180-class softmax | Unambiguous; high computation cost |

#### 5.2.2 Multi-Scale and Multi-Head Design

- Three-scale FPN (P3/P4/P5): YOLO26-OBB standard configuration, handling targets across size ranges;
- P2 augmentation (`yolo26-p2.yaml`): Smaller receptive field for small rotated target detection;
- End-to-end (`end2end`) mode: NMS-free one-to-one assignment, reducing NMS hyperparameter sensitivity.

### 5.3 Annotation Format Evolution

| Format | Representation | Advantages | Disadvantages |
|--------|---------------|-----------|--------------|
| 5-parameter $(x,y,w,h,\theta)$ | Direct regression | Compact | Angle ambiguity |
| 4 corner points $(x_1y_1 \ldots x_4y_4)$ | YOLO OBB format | Unambiguous, precise | 8 parameters |
| Polygon annotation | Arbitrary polygon | Most precise | Complex irregular shape handling |
| Radius + angle | Polar coordinates | Continuous angle | Less intuitive |

Ultralytics YOLO uses a **4-corner-point input with internal conversion to `xywhr`** strategy, balancing annotation flexibility with computational efficiency.

### 5.4 Loss Function Analysis

#### 5.4.1 L1 / Smooth L1 Angle Loss

$$
\mathcal{L}_{angle} = \text{Smooth L1}(\theta_{pred} - \theta_{gt})
$$

- **Strengths**: Simple implementation; smooth for small errors;
- **Weaknesses**: Completely ignores angle periodicity; loss value jumps abruptly at $0°/90°$ boundaries (boundary discontinuity); over-penalizes angle errors for symmetric (square-like) targets.

#### 5.4.2 IoU-Based Rotated Box Loss (RotatedIoU, GIoU)

Exact polygon intersection computation (Sutherland-Hodgman clipping) is expensive. Approximate methods include RotatedIoU and Generalized IoU for OBB, but gradient computation remains difficult through exact polygon operations.

#### 5.4.3 Probabilistic IoU (probiou)

Ultralytics YOLO26 adopts **probiou**, modeling rotated boxes as elliptical Gaussian distributions and using Bhattacharyya distance for IoU approximation:

$$
\text{BD}(\mathcal{N}_1, \mathcal{N}_2) = \frac{1}{4}(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)^T \boldsymbol{\Sigma}^{-1} (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2) + \frac{1}{2} \ln \frac{|\boldsymbol{\Sigma}|}{|\boldsymbol{\Sigma}_1|^{1/2}|\boldsymbol{\Sigma}_2|^{1/2}}
$$

$$
\text{probiou} = 1 - \sqrt{1 - e^{-\text{BD}}}
$$

The covariance matrix is derived from box width, height, and rotation angle:

```python
def _get_covariance_matrix(boxes):
    # boxes: (N, 5), xywhr format
    gbbs = torch.cat((boxes[:, 2:4].pow(2) / 12, boxes[:, 4:]), dim=-1)
    a, b, c = gbbs.split(1, dim=-1)
    cos, sin = c.cos(), c.sin()
    return a * cos**2 + b * sin**2, a * sin**2 + b * cos**2, (a - b) * cos * sin
```

- **Strengths**: Fully differentiable; stable gradients across continuous angle changes; jointly considers position, size, and rotation; avoids polygon intersection computation;
- **Weaknesses**: Gaussian approximation has limited accuracy for very elongated boxes; requires clamping for non-overlapping boxes where Bhattacharyya distance becomes large.

#### 5.4.4 Comparison Summary

| Angle Loss | Periodicity | Adaptive Weight | Form |
|-----------|-------------|----------------|------|
| Smooth L1 | ❌ | ❌ | L1 approximation |
| GWD (Wasserstein) | ✅ (implicit) | ❌ | Gaussian distance |
| CSL (classification) | ✅ (explicit) | ❌ | Classification loss |
| **OBB26 Sin-square** | ✅ (modular) | ✅ (aspect ratio) | Periodic smooth regression |

---

## 6. Typical Datasets and Evaluation Metrics

### 6.1 Public Rotated Object Detection Datasets

#### 6.1.1 DOTA Series

**DOTA (Dataset for Object Detection in Aerial Images)** is the most important benchmark in rotated object detection:

| Version | Categories | Images | Instances |
|---------|-----------|--------|-----------|
| DOTA-v1.0 | 15 | 2,806 | 188,282 |
| DOTA-v1.5 | 16 (+container crane) | 2,806 | 403,318 |
| DOTA-v2.0 | 18 (+airport, helipad) | 11,268 | 1,793,658 |

**Dataset characteristics**:
- Image resolutions from 800×800 to 20,000×20,000 pixels;
- Annotated with **arbitrary quadrilaterals (8 DoF 4-corner points)**;
- 15–18 categories including planes, ships, vehicles, buildings, and sports fields;
- Multi-scale challenges (extreme size variation within a single image).

In the Ultralytics project, DOTA-v1 is the standard pretraining dataset for YOLO26-OBB:

```bash
yolo obb train data=DOTAv1.yaml model=yolo26n-obb.pt epochs=100 imgsz=1024
```

#### 6.1.2 HRSC2016

**HRSC2016 (High-Resolution Ship Collections 2016)** focuses on ship detection:
- 702 high-resolution maritime/harbor images;
- 2,976 annotated instances with fine-grained ship categories (carriers, destroyers, etc.);
- Primarily elongated rotated targets, making it a critical benchmark for angle accuracy.

#### 6.1.3 UCAS-AOD

**UCAS-AOD (UCAS Aerial Object Detection)**:
- 1,510 aerial images, two categories: vehicles and aircraft;
- Relatively simple; commonly used for quick method validation.

#### 6.1.4 DOTA8 (Ultralytics Quick-Test Subset)

`DOTA8` is an 8-image subset (4 train + 4 val) extracted from DOTA-v1 for CI/CD and workflow validation. It does not represent production performance.

### 6.2 Evaluation Metrics

#### 6.2.1 mAP (Mean Average Precision)

The primary evaluation metric for rotated object detection, computed from precision-recall curves based on **rotated IoU** thresholds:

$$
\text{AP} = \int_0^1 p(r) \, dr
$$

$$
\text{mAP}_{50} = \frac{1}{C} \sum_{c=1}^C \text{AP}_c(\text{IoU} \geq 0.5)
$$

$$
\text{mAP}_{50:95} = \frac{1}{10} \sum_{t \in \{0.5, 0.55, \ldots, 0.95\}} \text{mAP}_{50}(t)
$$

On the DOTA evaluation server, predictions are submitted for online evaluation, and mAP@0.5 is the standard reported metric.

Accessing metrics in Ultralytics:

```python
metrics = model.val(data="dota8.yaml")
metrics.box.map     # mAP50-95
metrics.box.map50   # mAP50
metrics.box.map75   # mAP75
metrics.box.maps    # per-category mAP50-95 list
```

#### 6.2.2 Rotated IoU

Rotated box IoU is the fundamental overlap measure:

$$
\text{RotatedIoU} = \frac{|B_1 \cap B_2|}{|B_1 \cup B_2|}
$$

Exact computation requires polygon intersection algorithms (e.g., Sutherland-Hodgman). **Probabilistic IoU (probiou)** serves as an efficient differentiable approximation via Bhattacharyya distance between Gaussian distributions representing each box.

#### 6.2.3 Angle Error (MAE)

Some works additionally report mean absolute angle error (MAE) between predicted and ground-truth angles:

$$
\text{Angle MAE} = \frac{1}{N} \sum_{i=1}^{N} |\theta_i^{pred} - \theta_i^{gt}|
$$

Due to angle periodicity, differences must be computed after $\pi$-modular wrapping.

#### 6.2.4 Scale and Aspect-Ratio Grouped Metrics

For DOTA and similar datasets, some works report mAP grouped by target scale (small/medium/large) or aspect ratio to evaluate adaptability to diverse rotated targets.

---

## 7. Summary and Future Directions

### 7.1 Practical Effects of OBB and OBB26

OBB/OBB26 methods substantially outperform HBB in the following scenarios:

1. **High aspect-ratio elongated targets** (ships, aircraft): OBB tightly encloses the target, precise IoU;
2. **Dense packed rotated targets** (parking lots, warehouses): Accurate IoU keeps correct detections while suppressing duplicates;
3. **Arbitrary-orientation targets** (document text lines, industrial parts): Angle directly encodes orientation without post-processing inference.

OBB26 improvements over OBB:

- **No angle range constraint**: Allows the network to freely learn angle distributions without Sigmoid gradient vanishing;
- **Periodic loss**: Modular wrapping and sin-square function stably handle angle periodicity;
- **Adaptive aspect-ratio weighting**: Prevents near-square target angle loss from dominating training, improving angle accuracy for elongated targets.

### 7.2 Current Limitations

1. **Angle ambiguity**: For square targets ($w \approx h$), rotated box angle definition is non-unique; no method fully eliminates this;
2. **Numerical stability**: probiou requires clamping for very elongated or non-overlapping boxes;
3. **Computational overhead**: Rotated NMS (with probiou) is approximately 2–5× slower than axis-aligned box NMS;
4. **Annotation cost**: 4-corner rotated box annotation is more time-consuming than HBB; high-quality rotated box datasets remain scarce.

### 7.3 Future Directions

1. **Angle-free parameterization**: Gaussian representations (GWD, KFIoU) or polar coordinates to eliminate angle ambiguity entirely;
2. **End-to-end NMS-free detection**: DETR-class rotated object detection (e.g., ARS-DETR, Oriented DETR) to remove post-processing complexity;
3. **Foundation model adaptation**: Adapting vision foundation models (SAM, CLIP) for rotated object detection;
4. **Lightweight deployment**: Further reducing model parameters while maintaining rotated detection accuracy (OBB26-nano already achieves 2.7M parameters);
5. **Unified angle conventions**: Promoting standardized angle definitions across datasets and toolkits to reduce conversion overhead.

---

## 8. References

1. **DOTA Dataset**  
   Xia, G.-S. et al. "DOTA: A Large-scale Dataset for Object Detection in Aerial Images." *CVPR*, 2018.

2. **CSL (Circular Smooth Label)**  
   Yang, X. et al. "Arbitrary-Oriented Object Detection with Circular Smooth Label." *ECCV*, 2020.

3. **GWD (Gaussian Wasserstein Distance)**  
   Yang, X. et al. "Rethinking Rotated Object Detection with Gaussian Wasserstein Distance Loss." *ICML*, 2021.

4. **KFIoU**  
   Yang, X. et al. "The KFIoU Loss for Rotated Object Detection." *ICLR*, 2022.

5. **S2ANet**  
   Han, J. et al. "Align Deep Features for Oriented Object Detection." *IEEE TGRS*, 2021.

6. **Oriented R-CNN**  
   Xie, X. et al. "Oriented R-CNN for Object Detection." *ICCV*, 2021.

7. **probiou (Probabilistic IoU)**  
   Llerena, J. M. et al. "Gaussian Bounding Boxes and Probabilistic Intersection-over-Union for Object Detection." *arXiv:2106.06072*, 2021.

8. **Ultralytics YOLO26**  
   Ultralytics. "Ultralytics YOLO26 Documentation." https://docs.ultralytics.com/tasks/obb, 2024.

9. **RoI Transformer**  
   Ding, J. et al. "Learning RoI Transformer for Oriented Object Detection in Aerial Images." *CVPR*, 2019.

10. **HRSC2016**  
    Liu, Z. et al. "A High Resolution Optical Satellite Image Dataset for Ship Recognition and Some New Baselines." *ICPRAM*, 2017.
