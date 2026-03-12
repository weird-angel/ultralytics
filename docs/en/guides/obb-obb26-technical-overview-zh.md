---
comments: true
description: 旋转目标检测标签、OBB与OBB26原理及改进的系统性技术综述，涵盖HBB/OBB对比、角度规范差异、损失函数分析、典型数据集与评价指标，适用于学术论文场景。
keywords: OBB, OBB26, 旋转边界框, 旋转目标检测, HBB, 水平边界框, YOLO, 角度回归, probiou, DOTA, 目标检测, 旋转框
---

# 旋转目标检测标签、OBB与OBB26：技术综述

## 1. 引言

旋转目标检测（Oriented Object Detection）在通用计算机视觉应用场景中具有重要意义。传统目标检测方法通常采用**水平边界框（Horizontal Bounding Box，HBB）**对目标进行定位。然而，当目标出现倾斜、旋转或以任意方向分布时，HBB 无法精确拟合目标轮廓，会引入大量背景噪声，并在目标密集分布时产生严重的框重叠问题，从而降低检测精度。

为此，**旋转边界框（Oriented Bounding Box，OBB）**引入了额外的旋转角度参数，能够以任意方向旋转的矩形框精确包围目标。OBB 方法在航拍图像、车辆检测、文档分析及工业质检等领域均取得了显著的性能提升。

本文以 Ultralytics YOLO 项目的实际实现为基础，系统梳理旋转标签定义、OBB 与 HBB 的对比、OBB 与 OBB26 的原理、角度规范差异、损失函数分析、OBB26 相对于已发表工作的改进与优缺点，并结合典型数据集与评价指标进行说明，以期为相关学术论文的撰写提供参考。

---

## 2. 旋转图片标签定义格式

### 2.1 水平边界框（HBB）

水平边界框是目标检测中最常见的标注格式：

$$
\text{HBB} = (c_x, c_y, w, h)
$$

其中 $c_x, c_y$ 为目标中心坐标，$w, h$ 分别为宽度和高度。HBB 始终与坐标轴对齐（轴对齐边界框，AABB），不包含任何方向信息。

**HBB 的主要局限性：**

- 对倾斜目标，轴对齐框会包含大量背景区域，引入噪声干扰；
- 密集排列的旋转目标（如停车场车辆、港口舰船）会产生大面积重叠框，严重影响 NMS 效果；
- 完全缺失方向信息，限制了后续需要感知目标朝向的下游任务。

### 2.2 旋转边界框（OBB）

OBB 在 HBB 基础上增加旋转角度参数：

$$
\text{OBB} = (c_x, c_y, w, h, \theta)
$$

其中 $\theta$ 为目标框相对于参考方向的旋转角度。OBB 能够精确拟合任意朝向的矩形目标，最小化外接面积，显著提升检测精度与 NMS 质量。

**OBB 与 HBB 的对比：**

| 特性 | HBB | OBB |
|------|-----|-----|
| 旋转目标拟合 | ❌ 框面积大，含大量背景 | ✅ 精确拟合，最小外接矩形 |
| 密集目标重叠 | ❌ IoU 偏大，NMS 误删 | ✅ IoU 精准，抑制准确 |
| 朝向信息 | ❌ 完全缺失 | ✅ 直接编码旋转角度 |
| 标注复杂度 | ✅ 简单（4 个参数） | ❌ 较复杂（5 参数 / 8 点） |
| 推理复杂度 | ✅ 标准 NMS | ❌ 需要旋转框 NMS |

### 2.3 OBB 标注格式

在 Ultralytics YOLO 项目中，OBB 的**标注格式**（标签文件）采用 **4 个归一化角点坐标**：

```
class_index  x1 y1  x2 y2  x3 y3  x4 y4
```

其中 $(x_i, y_i)$ 为旋转框四个顶点的归一化坐标（取值范围 $[0, 1]$），按顺序排列。示例：

```
0  0.780811 0.743961  0.782371 0.74686  0.777691 0.752174  0.776131 0.749758
```

**内部处理格式（`xywhr`）**：在计算损失和输出预测时，YOLO 将 4 角点格式转换为 `xywhr`：

$$
\text{xywhr} = (c_x, c_y, w, h, r)
$$

其中 $r$ 为旋转角度（弧度制）。该格式便于参数化回归，并与旋转 IoU 计算保持一致。

### 2.4 OBB26 标签表示

OBB26 是 Ultralytics YOLO26 系列中用于旋转目标检测的检测头变体。从标注格式角度看，OBB26 与 OBB 使用**完全相同的标注格式**（`class_index x1 y1 x2 y2 x3 y3 x4 y4`）。两者的核心差异在于**网络角度预测的处理方式**，详见第 4 节。

---

## 3. OBB 角度规范差异与 YOLO 的结合方式

### 3.1 角度定义规范的多样性

旋转边界框中的角度 $\theta$ 在不同数据集、工具库和研究工作中存在多种定义规范，差异较为显著。

#### 3.1.1 OpenCV 规范

`cv2.minAreaRect` 返回 $\theta \in [-90°, 0°)$：
- 定义为从正 x 轴（水平向右）**顺时针**旋转到返回矩形的**"width"边**所经过的角度；**"width"边不一定是最长边**，OpenCV 对此没有约束；
- 顺时针方向为负；水平放置的框返回 $\theta = 0°$；旋转至竖直方向时 $\theta$ 趋向 $-90°$。

#### 3.1.2 短边逆时针规范

部分数据集（包括 DOTA）和研究工作采用**以短边为基、逆时针旋转**的规范：
- $\theta \in [0°, 90°)$，定义为短边与水平轴形成的逆时针角度；
- 当宽高相等时定义不唯一（需额外约定）。

#### 3.1.3 YOLO Sigmoid 映射规范（原始 OBB 检测头）

在 Ultralytics YOLO 的 OBB 检测头中，角度以弧度表示，通过 Sigmoid 激活约束到特定范围：

$$
\theta = (\text{sigmoid}(x) - 0.25) \times \pi \in \left[-\frac{\pi}{4}, \frac{3\pi}{4}\right]
$$

该设计将角度范围映射到 $[-45°, 135°)$。$-0.25\pi$ 的偏置将典型的 $[0°, \pi)$ 范围向左偏移，以更好地覆盖常见旋转角度分布。

!!! note "YOLO OBB 角度约束"
    根据项目文档说明，OBB 角度被约束在 **0–90 度（不含 90 度）** 范围内，角度 ≥ 90° 的情况不被直接支持，标注时应避免此类情况。

#### 3.1.4 OBB26 无约束规范

OBB26 检测头输出**原始、无约束的角度预测值**，不施加 Sigmoid 激活。角度周期性由损失函数显式处理（详见第 4.2 节）。

#### 3.1.5 规范对比汇总

| 规范类型 | 角度范围 | 周期性处理 | 边界处梯度 |
|---------|---------|-----------|----------|
| OpenCV | $[-90°, 0°)$ | 边界跳跃 | 不连续 |
| 短边逆时针 | $[0°, 90°)$ | 边界跳跃 | 不连续 |
| YOLO Sigmoid | $[-45°, 135°)$ | Sigmoid 抑制边界 | 接近边界时趋零 |
| OBB26 无约束 | 原始预测值 | 损失层模运算 | 全程平滑 |

**核心挑战**：矩形框具有 $180°$ 旋转对称性（角度 $\theta$ 与 $\theta + 180°$ 对应完全相同的框），导致回归目标存在"多解性"歧义。这一不一致性在梯度反传时引起训练不稳定，是大多数角度感知损失函数设计的根本动机。

#### 3.1.6 为什么 YOLO 的 xywhr 角度范围是 [-45°, 135°) 而不是 [-90°, 0°)?

YOLO 的 `xyxyxyxy2xywhr`（`ultralytics/utils/ops.py`）函数将 4 角点格式转换为 `xywhr`，其中依次执行以下步骤，导致最终角度范围从 OpenCV 的 $[-90°, 0°)$ 变为 $[-45°, 135°)$。

**第 1 步 — 调用 OpenCV**

```python
(cx, cy), (w, h), angle = cv2.minAreaRect(pts)
theta = angle / 180 * np.pi   # 度 → 弧度；theta ∈ [-π/2, 0)
```

OpenCV 返回 `angle ∈ [-90°, 0°)`，其对应的 `w` 是 OpenCV 内部定义的"width"边，**可能是短边也可能是长边**，没有保证 `w >= h`。

**第 2 步 — 强制 w ≥ h（长边始终称为 w）**

```python
if w < h:
    w, h = h, w
    theta += np.pi / 2   # 现在参考的是长边，其方向比短边多 90°
```

当 OpenCV 的 `w < h` 时，"width"边实际上是短边。YOLO 交换 `w` 和 `h`，使 `w` 始终对应长边（物体的主要延伸方向）。几何上，长边与短边方向相差 90°，因此 `theta += π/2`。

交换后各情形下 theta 的范围：

| 情形 | OpenCV 输出 | 交换后 theta |
|------|------------|------------|
| `w ≥ h`（无需交换） | `w ≥ h`，`angle ∈ [-90°, 0°)` | `theta ∈ [-π/2, 0)` |
| `w < h`（需要交换） | `w < h`，`angle ∈ [-90°, 0°)` | `theta ∈ [0, π/2)` |

**第 3 步 — 归一化到 [-π/4, 3π/4)**

```python
while theta >= 3 * np.pi / 4:
    theta -= np.pi        # 从 135° 以上向下折叠
while theta < -np.pi / 4:
    theta += np.pi        # 从 -45° 以下向上折叠
```

对第 2 步中各范围应用归一化循环的结果：

| 归一化前 theta | 归一化后结果 |
|--------------|------------|
| `[-π/2, -π/4)` 即 `[-90°, -45°)`（无交换，角度较陡） | `+π` → `[π/2, 3π/4)` 即 `[90°, 135°)` |
| `[-π/4, 0)` 即 `[-45°, 0°)`（无交换，角度较浅） | 不变，保持 `[-π/4, 0)` |
| `[0, π/2)` 即 `[0°, 90°)`（交换后） | 不变，保持 `[0, π/2)` |

合并后的最终范围：$[-\pi/4,\ 0) \cup [0,\ \pi/2) \cup [\pi/2,\ 3\pi/4) = [-\pi/4,\ 3\pi/4)$，即 **[-45°, 135°)**。

**核心区别总结**

| 项目 | OpenCV 角度 | YOLO xywhr 角度 |
|------|-----------|---------------|
| 参考边 | OpenCV 内部的"width"边（不保证是长边） | 始终是**长边**（`w ≥ h`） |
| 角度范围 | $[-90°, 0°)$ | $[-45°, 135°)$ |
| 范围变化原因 | — | 长边方向 = 短边方向 ± 90°，加上归一化循环 |

OBB 检测头的 Sigmoid 公式 `(sigmoid(x) - 0.25) × π` 正是为了与这一范围严格对齐：

$$
\sigma(x) \in (0,\ 1) \;\Longrightarrow\; (\sigma(x) - 0.25)\pi \in \left(-\frac{\pi}{4},\ \frac{3\pi}{4}\right)
$$

### 3.2 OBB 在 YOLO 中的实现要点

#### 3.2.1 标签转换：4 角点格式 → xywhr（`xyxyxyxy2xywhr`）

从标注 4 角点格式到内部 `xywhr` 表示的完整转换代码位于 `ultralytics/utils/ops.py`：

```python
def xyxyxyxy2xywhr(x):
    """将 [xy1,xy2,xy3,xy4] (N,8) 转换为 [cx,cy,w,h,theta] (N,5)，theta ∈ [-pi/4, 3pi/4)"""
    ...
    for pts in points:
        # 第 1 步：调用 OpenCV，返回 angle ∈ [-90°, 0°)，w 不保证 >= h
        (cx, cy), (w, h), angle = cv2.minAreaRect(pts)

        # 第 2 步：度 → 弧度
        theta = angle / 180 * np.pi                       # theta ∈ [-π/2, 0)

        # 第 3 步：强制 w >= h（长边始终称为 w）
        if w < h:
            w, h = h, w
            theta += np.pi / 2   # 长边方向 = 短边方向 + 90°

        # 第 4 步：归一化到 [-π/4, 3π/4)
        while theta >= 3 * np.pi / 4:
            theta -= np.pi
        while theta < -np.pi / 4:
            theta += np.pi

        rboxes.append([cx, cy, w, h, theta])
```

这正是 YOLO xywhr 角度范围为 **[-45°, 135°)** 而非 OpenCV 的 [-90°, 0°) 的直接原因——详见第 3.1.6 节的逐步推导。

#### 3.2.2 检测头结构

在 `ultralytics/nn/modules/head.py` 中，`OBB` 类继承自 `Detect`，增加了专用的角度预测分支：

```python
class OBB(Detect):
    def __init__(self, nc=80, ne=1, reg_max=16, end2end=False, ch=()):
        super().__init__(nc, reg_max, end2end, ch)
        self.ne = ne  # 角度参数数量（默认为 1）
        c4 = max(ch[0] // 4, self.ne)
        # 专用角度预测卷积分支
        self.cv4 = nn.ModuleList(
            nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.ne, 1))
            for x in ch
        )
```

带 Sigmoid 激活的角度预测（原始 OBB）：

```python
def forward_head(self, x, box_head, cls_head, angle_head):
    # ...
    angle = (angle.sigmoid() - 0.25) * math.pi  # 映射到 [-π/4, 3π/4]
    preds["angle"] = angle
```

#### 3.2.3 旋转框解码（`dist2rbox`）

YOLO 采用**分布式焦点损失（DFL）**框架，将框参数预测为从锚点出发的距离分布（ltrb 格式）。针对旋转框，通过 `dist2rbox` 函数完成解码：

```python
def dist2rbox(pred_dist, pred_angle, anchor_points, dim=-1):
    lt, rb = pred_dist.split(2, dim=dim)
    cos, sin = torch.cos(pred_angle), torch.sin(pred_angle)
    xf, yf = ((rb - lt) / 2).split(1, dim=dim)
    x, y = xf * cos - yf * sin, xf * sin + yf * cos
    xy = torch.cat([x, y], dim=dim) + anchor_points
    return torch.cat([xy, lt + rb], dim=dim)
```

该函数将预测的 ltrb 距离与角度解码为旋转框的中心坐标和宽高（`xywh` 格式）。

#### 3.2.4 NMS 后处理

旋转框的非极大值抑制（NMS）采用**概率 IoU（probiou）**进行高效的旋转框 IoU 计算，避免了代价高昂的多边形交集计算：

```python
# 旋转 NMS 使用 batch_probiou 计算 IoU
i = TorchNMS.fast_nms(boxes, scores, iou_thres, iou_func=batch_probiou)
```

---

## 4. OBB26 的原理与改进

### 4.1 OBB 与 OBB26 的核心区别

`OBB26` 类（`ultralytics/nn/modules/head.py`）在继承 `OBB` 的基础上，实施了关键改进：**移除角度预测上的 Sigmoid 激活函数**：

```python
class OBB26(OBB):
    def forward_head(self, x, box_head, cls_head, angle_head):
        preds = Detect.forward_head(self, x, box_head, cls_head)
        if angle_head is not None:
            bs = x[0].shape[0]
            angle = torch.cat(
                [angle_head[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2
            )  # 原始角度预测——无 Sigmoid 约束
            preds["angle"] = angle
        return preds
```

| 特性 | OBB（原始） | OBB26（改进） |
|------|------------|--------------|
| 角度激活函数 | Sigmoid → $[-\pi/4, 3\pi/4]$ | 无（无约束） |
| 角度范围 | 固定在 $[-45°, 135°)$ | 网络自由学习 |
| 周期性处理 | Sigmoid 抑制边界跳跃 | 损失层模运算 |
| 所属架构 | YOLO 早期 OBB | YOLO26 系列 |

**设计动机**：Sigmoid 约束虽然限制了角度回归的范围，但在接近边界（$-45°$ 或 $135°$）时 Sigmoid 梯度趋近于零，导致梯度消失和训练不稳定。OBB26 移除该约束，配合周期感知损失函数实现更稳定的角度回归。

### 4.2 周期感知角度损失（`calculate_angle_loss`）

OBB26 采用**周期感知的 Sin-square 角度损失**，并结合长宽比自适应权重：

```python
def calculate_angle_loss(self, pred_bboxes, target_bboxes, fg_mask, weight,
                          target_scores_sum, lambda_val=3):
    w_gt = target_bboxes[..., 2]
    h_gt = target_bboxes[..., 3]
    pred_theta = pred_bboxes[..., 4]
    target_theta = target_bboxes[..., 4]

    # 长宽比权重：近似正方形的框对角度不敏感，权重低
    log_ar = torch.log((w_gt + 1e-9) / (h_gt + 1e-9))
    scale_weight = torch.exp(-(log_ar**2) / (lambda_val**2))

    # 角度差的模运算包装（π 周期）
    delta_theta = pred_theta - target_theta
    delta_theta_wrapped = delta_theta - torch.round(delta_theta / math.pi) * math.pi

    # Sin-square 角度损失
    ang_loss = torch.sin(2 * delta_theta_wrapped[fg_mask]) ** 2
    ang_loss = scale_weight[fg_mask] * ang_loss * weight
    return ang_loss.sum() / target_scores_sum
```

**关键设计要点：**

1. **模运算包装（$\pi$ 取模）**：`delta_theta_wrapped = delta_theta - round(delta_theta/π) * π` 将角度差映射到 $[-\pi/2, \pi/2)$，消除矩形框 $180°$ 对称性引起的梯度方向混乱。

2. **Sin-square 损失**：`sin(2Δθ)²` 是在 $\Delta\theta = 0$ 处有零点、在 $\pm\pi/4$ 处达到最大值的光滑周期函数，在零点附近具有二阶平滑性，避免了 L1 不可微问题和 L2 在角度回归中方差过大的问题。

3. **长宽比自适应权重**：`scale_weight = exp(-(log(w/h))²/λ²)` 降低近似正方形目标（角度歧义影响小）的角度损失贡献，对细长目标则权重接近 1.0，有效提升高长宽比目标的角度预测精度。

**公式汇总：**

$$
\mathcal{L}_{angle} = \frac{1}{N_+} \sum_{i \in +} w_i \cdot \sin^2(2\Delta\theta_{\text{wrapped},i})
$$

$$
\Delta\theta_{\text{wrapped}} = \Delta\theta - \text{round}\!\left(\frac{\Delta\theta}{\pi}\right) \cdot \pi, \qquad
w_i = \exp\!\left(-\frac{(\ln(w_{gt}/h_{gt}))^2}{\lambda^2}\right)
$$

### 4.3 YOLO26-OBB 架构

在 YOLO26-obb 模型配置文件（`ultralytics/cfg/models/26/yolo26-obb.yaml`）中：

```yaml
# 检测头采用 OBB26
- [[16, 19, 22], 1, OBB26, [nc, 1]]  # OBB26(P3, P4, P5)
```

模型采用 `end2end: True` 与轻量化 `reg_max: 1` 的 DFL 设计，降低了多 bin DFL 的计算开销，并通过端到端无 NMS 训练（one-to-one 匹配）提升推理速度。

| 尺度 | 参数量 | GFLOPs |
|------|--------|--------|
| nano (n) | 2.7M | 16.9 |
| small (s) | 10.6M | 63.5 |
| medium (m) | 23.6M | 211.9 |
| large (l) | 28.0M | 259.0 |
| extra-large (x) | 62.8M | 578.9 |

---

## 5. 已发表方案对比：改进点与优缺点分析

### 5.1 旋转目标检测技术路线演进

#### 5.1.1 基于锚框的早期方法

**代表工作**：R2CNN、RRPN、RoI Transformer

- **优点**：基于成熟的 Faster R-CNN 框架，检测精度较高；
- **缺点**：角度直接回归需要额外约束；存在严重的**边界不连续问题**（$0°/90°$ 边界处数值跳变导致梯度不稳定）；两阶段推理速度较慢。

#### 5.1.2 角度编码改进方法

**CSL（Circular Smooth Label，ECCV 2020）**：将角度回归转化为以 180 个软标签区间为目标的分类问题，采用环形平滑标签：
- 优点：彻底消除边界跳跃，预测稳定；
- 缺点：180 通道分类头参数量大；精度依赖区间分辨率。

**DCL（CVPR 2021）**：利用 Gray 编码压缩 CSL 分类头：
- 优点：比 CSL 更高效；
- 缺点：编解码较复杂；边界处仍有轻微残余不连续。

**GWD（CVPR 2021）**：将旋转框建模为二维高斯分布，以 Wasserstein 距离作为损失：
- 优点：完全绕过显式角度回归，对角度周期性鲁棒；
- 缺点：高斯近似不完全精确；对近似正方形目标效果下降。

**KFIoU（ICLR 2022）**：基于卡尔曼滤波的 IoU 近似，在 GWD 基础上提供更精确的重叠度估计：
- 优点：尺度一致性好，DOTA 性能优异；
- 缺点：实现复杂，引入额外超参数。

#### 5.1.3 基于 YOLO 的单阶段方法

**S2ANet（TGRS 2021）**：引入特征对齐模块（Feature Alignment Module）和轴对齐 NMS：
- 优点：检测精度高，DOTA 表现出色；
- 缺点：结构复杂，推理较慢。

**Ultralytics YOLO26-OBB / OBB26**：将 OBB26 检测头集成到轻量级单阶段 YOLO 中，端到端训练：
- 优点：速度快，部署简便，与 YOLO 生态完全兼容；
- 缺点：对极端角度（接近 $\pm45°$ 或 $135°$ 边界）的处理依赖损失函数而非结构保证。

### 5.2 结构变动方向

#### 5.2.1 角度参数化方式对比

| 方法 | 参数化形式 | 特点 |
|------|-----------|------|
| 直接回归 | $\theta \in [0°, 90°)$ | 简单，但边界问题严重 |
| Sigmoid 映射（OBB） | $\sigma(x) \times \pi$ | 有界范围，远离边界时梯度稳定 |
| 无约束原始预测（OBB26） | 原始 logit + 损失包装 | 灵活，需配合周期损失 |
| 正余弦分解 | $(\cos 2\theta, \sin 2\theta)$ | 连续且无歧义；角度参数翻倍 |
| 软分类（CSL） | 180 类 softmax | 无歧义；计算开销大 |

#### 5.2.2 多尺度与多头设计

- 三尺度 FPN（P3/P4/P5）：YOLO26-OBB 标准配置，适应不同尺寸的旋转目标；
- P2 增强（`yolo26-p2.yaml`）：更小感受野，适合小目标旋转检测；
- 端到端（`end2end`）模式：无 NMS 的 one-to-one 匹配，降低 NMS 超参数敏感性。

### 5.3 标注方式的演进

| 标注方式 | 表示形式 | 优点 | 缺点 |
|---------|---------|------|------|
| 五参数 $(x,y,w,h,\theta)$ | 直接回归 | 紧凑 | 角度歧义 |
| 四角点 $(x_1y_1 \ldots x_4y_4)$ | YOLO OBB 格式 | 无歧义，精确 | 8 个参数 |
| 多边形标注 | 任意多边形 | 最精确 | 不规则形状处理复杂 |
| 半径 + 角度 | 极坐标 | 角度连续 | 不够直观 |

Ultralytics YOLO 采用**四角点输入 + 内部转换为 `xywhr`** 的策略，在标注灵活性与计算效率之间取得平衡。

### 5.4 损失函数的优劣分析

#### 5.4.1 L1 / Smooth L1 角度损失

$$
\mathcal{L}_{angle} = \text{Smooth L1}(\theta_{pred} - \theta_{gt})
$$

- **优点**：实现简单；对小误差平滑；
- **缺点**：完全忽略角度周期性；在 $0°/90°$ 边界处损失值突变（边界不连续问题）；对对称（近似正方形）目标的角度误差惩罚过重。

#### 5.4.2 基于 IoU 的旋转框损失（RotatedIoU、GIoU）

精确的多边形交集计算（Sutherland-Hodgman 裁剪算法）代价高昂。RotatedIoU 和面向 OBB 的 GIoU 等近似方法，在精确多边形运算上微分计算仍较困难。

#### 5.4.3 概率 IoU（probiou）

Ultralytics YOLO26 采用 **probiou**，将旋转框建模为椭圆高斯分布，通过 Bhattacharyya 距离近似 IoU：

$$
\text{BD}(\mathcal{N}_1, \mathcal{N}_2) = \frac{1}{4}(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)^T \boldsymbol{\Sigma}^{-1} (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2) + \frac{1}{2} \ln \frac{|\boldsymbol{\Sigma}|}{|\boldsymbol{\Sigma}_1|^{1/2}|\boldsymbol{\Sigma}_2|^{1/2}}
$$

$$
\text{probiou} = 1 - \sqrt{1 - e^{-\text{BD}}}
$$

协方差矩阵由旋转框的宽高和旋转角推导：

```python
def _get_covariance_matrix(boxes):
    # boxes: (N, 5), xywhr 格式
    gbbs = torch.cat((boxes[:, 2:4].pow(2) / 12, boxes[:, 4:]), dim=-1)
    a, b, c = gbbs.split(1, dim=-1)
    cos, sin = c.cos(), c.sin()
    return a * cos**2 + b * sin**2, a * sin**2 + b * cos**2, (a - b) * cos * sin
```

- **优点**：完全可微分；角度连续变化时梯度稳定；同时考虑位置、尺寸和旋转；避免多边形交集计算；
- **缺点**：高斯近似对极细长框的精度有限；完全不重叠的框 Bhattacharyya 距离较大，需 clamp 处理。

#### 5.4.4 损失函数对比汇总

| 角度损失 | 周期性处理 | 自适应权重 | 形式 |
|---------|-----------|-----------|------|
| Smooth L1 | ❌ | ❌ | L1 近似 |
| GWD（Wasserstein） | ✅（隐式） | ❌ | 高斯距离 |
| CSL（分类） | ✅（显式） | ❌ | 分类损失 |
| **OBB26 Sin-square** | ✅（模运算） | ✅（长宽比） | 周期光滑回归 |

---

## 6. 典型数据集与评价指标

### 6.1 公开旋转目标检测数据集

#### 6.1.1 DOTA 系列

**DOTA（Dataset for Object Detection in Aerial Images）** 是旋转目标检测领域最重要的基准数据集：

| 版本 | 类别数 | 图像数 | 标注数 |
|------|--------|--------|--------|
| DOTA-v1.0 | 15 | 2,806 | 188,282 |
| DOTA-v1.5 | 16（+集装箱吊车） | 2,806 | 403,318 |
| DOTA-v2.0 | 18（+机场/直升机坪） | 11,268 | 1,793,658 |

**数据集特点**：
- 图像分辨率从 800×800 到 20,000×20,000 像素不等；
- 采用**任意四边形（8 自由度四角点）标注**；
- 涵盖飞机、舰船、车辆、建筑、操场等 15–18 类目标；
- 包含多尺度挑战（同一图像中目标尺寸差异极大）。

在 Ultralytics 项目中，DOTA-v1 是 YOLO26-OBB 的标准预训练数据集：

```bash
yolo obb train data=DOTAv1.yaml model=yolo26n-obb.pt epochs=100 imgsz=1024
```

#### 6.1.2 HRSC2016

**HRSC2016（High-Resolution Ship Collections 2016）** 专注于舰船检测：
- 702 张高分辨率海洋/港口图像；
- 2,976 个标注实例，涵盖航母、驱逐舰等细粒度舰船类别；
- 以细长旋转目标为主，是评测角度预测精度的重要基准。

#### 6.1.3 UCAS-AOD

**UCAS-AOD（UCAS Aerial Object Detection）**：
- 1,510 张航拍图像，车辆和飞机两个类别；
- 相对简单，常用于旋转检测方法的快速验证。

#### 6.1.4 DOTA8（Ultralytics 快速测试子集）

`DOTA8` 是从 DOTA-v1 中抽取的 8 张图像子集（4 张训练 + 4 张验证），用于 CI/CD 流程验证，不代表实际生产性能。

### 6.2 评价指标

#### 6.2.1 mAP（Mean Average Precision）

旋转目标检测的主要评价指标，基于**旋转 IoU** 阈值计算精度-召回曲线下的面积：

$$
\text{AP} = \int_0^1 p(r) \, dr
$$

$$
\text{mAP}_{50} = \frac{1}{C} \sum_{c=1}^C \text{AP}_c(\text{IoU} \geq 0.5)
$$

$$
\text{mAP}_{50:95} = \frac{1}{10} \sum_{t \in \{0.5, 0.55, \ldots, 0.95\}} \text{mAP}_{50}(t)
$$

在 DOTA 评测服务器上，需提交测试集预测结果进行在线评估，标准指标为 mAP@0.5。

在 Ultralytics 中访问验证指标：

```python
metrics = model.val(data="dota8.yaml")
metrics.box.map     # mAP50-95
metrics.box.map50   # mAP50
metrics.box.map75   # mAP75
metrics.box.maps    # 各类别的 mAP50-95 列表
```

#### 6.2.2 旋转 IoU（Rotated IoU）

旋转框 IoU 是衡量预测框与真值框重叠程度的基础度量：

$$
\text{RotatedIoU} = \frac{|B_1 \cap B_2|}{|B_1 \cup B_2|}
$$

精确计算需要多边形交集算法（如 Sutherland-Hodgman 算法）。**概率 IoU（probiou）**通过对两个旋转框所代表的高斯分布计算 Bhattacharyya 距离，提供高效的可微分近似。

#### 6.2.3 角度误差（Angle MAE）

部分工作额外报告预测角度与真值角度之间的平均绝对误差（MAE）：

$$
\text{Angle MAE} = \frac{1}{N} \sum_{i=1}^{N} |\theta_i^{pred} - \theta_i^{gt}|
$$

由于角度具有周期性，计算前需进行 $\pi$ 模运算包装。

#### 6.2.4 按尺度和长宽比分组的指标

针对 DOTA 等数据集，部分工作按目标尺度（小/中/大）或长宽比分组报告 mAP，以评估方法对不同尺寸旋转目标的适应能力。

---

## 7. 总结与展望

### 7.1 OBB 与 OBB26 的实践效果

OBB/OBB26 方法在以下场景中较传统 HBB 具有显著优势：

1. **高长宽比细长目标**（舰船、飞机）：OBB 紧密包围目标，IoU 计算精确；
2. **密集排列的旋转目标**（停车场、仓储货物）：精确 IoU 在保留正确检测的同时有效抑制重复框；
3. **任意方向目标**（文档文字行、工业零件）：角度参数直接描述朝向，无需后处理推断。

OBB26 相对于 OBB 的主要提升：

- **无角度范围约束**：允许网络自由学习角度分布，避免 Sigmoid 梯度消失；
- **周期感知损失**：模运算包装与 Sin-square 函数稳定处理角度周期性；
- **自适应长宽比权重**：防止近正方形目标的角度损失主导训练，提升细长目标的角度精度。

### 7.2 现有局限性

1. **角度歧义**：对于正方形目标（$w \approx h$），旋转框角度定义不唯一；任何方法均无法彻底消除此问题；
2. **数值稳定性**：probiou 对极细长框或完全不重叠框需要 clamp 处理；
3. **计算开销**：旋转框 NMS（含 probiou 计算）比轴对齐框 NMS 慢约 2–5 倍；
4. **标注成本**：4 角点旋转框标注比 HBB 更耗时；高质量旋转框数据集仍较稀缺。

### 7.3 未来发展方向

1. **无角度参数化**：基于高斯表示（GWD、KFIoU）或极坐标，从根本上消除角度歧义；
2. **端到端无 NMS 检测**：DETR 类旋转目标检测（如 ARS-DETR、Oriented DETR）消除后处理复杂度；
3. **视觉基础模型适配**：将 SAM、CLIP 等视觉基础模型迁移到旋转目标检测任务；
4. **轻量化部署**：在保持旋转检测精度的前提下进一步压缩模型参数（OBB26-nano 已实现 2.7M 参数）；
5. **角度规范统一**：推动数据集与工具库采用统一角度定义，降低迁移与转换开销。

---

## 8. 参考文献

1. **DOTA 数据集**  
   Xia, G.-S. 等. "DOTA: A Large-scale Dataset for Object Detection in Aerial Images." *CVPR*, 2018.

2. **CSL（Circular Smooth Label）**  
   Yang, X. 等. "Arbitrary-Oriented Object Detection with Circular Smooth Label." *ECCV*, 2020.

3. **GWD（Gaussian Wasserstein Distance）**  
   Yang, X. 等. "Rethinking Rotated Object Detection with Gaussian Wasserstein Distance Loss." *ICML*, 2021.

4. **KFIoU**  
   Yang, X. 等. "The KFIoU Loss for Rotated Object Detection." *ICLR*, 2022.

5. **S2ANet**  
   Han, J. 等. "Align Deep Features for Oriented Object Detection." *IEEE TGRS*, 2021.

6. **Oriented R-CNN**  
   Xie, X. 等. "Oriented R-CNN for Object Detection." *ICCV*, 2021.

7. **probiou（概率 IoU）**  
   Llerena, J. M. 等. "Gaussian Bounding Boxes and Probabilistic Intersection-over-Union for Object Detection." *arXiv:2106.06072*, 2021.

8. **Ultralytics YOLO26**  
   Ultralytics. "Ultralytics YOLO26 Documentation." https://docs.ultralytics.com/tasks/obb, 2024.

9. **RoI Transformer**  
   Ding, J. 等. "Learning RoI Transformer for Oriented Object Detection in Aerial Images." *CVPR*, 2019.

10. **HRSC2016**  
    Liu, Z. 等. "A High Resolution Optical Satellite Image Dataset for Ship Recognition and Some New Baselines." *ICPRAM*, 2017.
