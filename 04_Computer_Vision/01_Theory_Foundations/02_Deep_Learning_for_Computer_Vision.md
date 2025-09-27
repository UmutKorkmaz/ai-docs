# Deep Learning for Computer Vision

## Overview
Deep learning has revolutionized computer vision by enabling systems to learn hierarchical representations directly from data. This section covers the theoretical foundations, architectures, and mathematical principles behind deep learning models for visual understanding.

## 1. Convolutional Neural Networks (CNNs)

### 1.1 CNN Architecture Fundamentals

#### 1.1.1 Core Components
**Convolutional Layers:**
The fundamental operation that applies learnable filters to input data:

$$y_{i,j} = \sum_{m=0}^{M-1} \sum_{n=0}^{N-1} x_{i+m,j+n} \cdot w_{m,n} + b$$

Where:
- $y_{i,j}$ is the output at position $(i,j)$
- $x$ is the input patch
- $w$ is the kernel weights
- $b$ is the bias term
- $M \times N$ is the kernel size

**Pooling Layers:**
Reduce spatial dimensions while preserving important features:

- **Max Pooling**: $y_{i,j} = \max_{m,n \in \text{pool}} x_{i+m,j+n}$
- **Average Pooling**: $y_{i,j} = \frac{1}{|\text{pool}|}\sum_{m,n \in \text{pool}} x_{i+m,j+n}$

**Activation Functions:**
- **ReLU**: $f(x) = \max(0, x)$
- **Leaky ReLU**: $f(x) = \max(0.01x, x)$
- **ELU**: $f(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha(e^x - 1) & \text{if } x \leq 0 \end{cases}$

#### 1.1.2 CNN Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BasicCNN(nn.Module):
    """Basic CNN architecture for image classification"""
    def __init__(self, num_classes=10):
        super(BasicCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Dropout
        self.dropout = nn.Dropout(0.5)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x

class ResidualBlock(nn.Module):
    """Residual block for ResNet-style architectures"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += residual
        out = F.relu(out)

        return out
```

### 1.2 Advanced CNN Architectures

#### 1.2.1 ResNet (Residual Networks)

```python
class ResNet(nn.Module):
    """ResNet implementation with configurable depth"""
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channels = 64

        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, channels, stride))
        self.in_channels = channels * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet18(num_classes=1000):
    """ResNet-18 architecture"""
    return ResNet(ResidualBlock, [2, 2, 2, 2], num_classes)

def resnet50(num_classes=1000):
    """ResNet-50 architecture with bottleneck blocks"""
    class BottleneckBlock(nn.Module):
        expansion = 4

        def __init__(self, in_channels, out_channels, stride=1):
            super(BottleneckBlock, self).__init__()

            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)

            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                  stride=stride, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)

            self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                                  kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_channels != out_channels * self.expansion:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels * self.expansion,
                             kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels * self.expansion)
                )

        def forward(self, x):
            residual = self.shortcut(x)

            out = F.relu(self.bn1(self.conv1(x)))
            out = F.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))

            out += residual
            out = F.relu(out)

            return out

    return ResNet(BottleneckBlock, [3, 4, 6, 3], num_classes)
```

#### 1.2.2 EfficientNet

```python
import math

class Swish(nn.Module):
    """Swish activation function"""
    def forward(self, x):
        return x * torch.sigmoid(x)

class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block"""
    def __init__(self, channels, reduction_ratio=4):
        super(SqueezeExcitation, self).__init__()

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()

        # Squeeze
        y = self.squeeze(x).view(b, c)

        # Excitation
        y = self.excitation(y).view(b, c, 1, 1)

        # Scale
        return x * y

class MBConvBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck block"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio):
        super(MBConvBlock, self).__init__()

        self.use_residual = stride == 1 and in_channels == out_channels
        hidden_dim = in_channels * expand_ratio

        # Expansion phase
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            Swish()
        ) if expand_ratio != 1 else nn.Identity()

        # Depthwise convolution
        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride,
                     kernel_size//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            Swish()
        )

        # Squeeze-and-excitation
        self.se = SqueezeExcitation(hidden_dim, 1/se_ratio) if se_ratio > 0 else nn.Identity()

        # Projection phase
        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        residual = x

        x = self.expand(x)
        x = self.depthwise(x)
        x = self.se(x)
        x = self.project(x)

        if self.use_residual:
            x += residual

        return x

def efficientnet_b0(num_classes=1000):
    """EfficientNet-B0 implementation"""
    # Architecture configuration for EfficientNet-B0
    config = [
        # in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio
        (32, 16, 3, 1, 1, 0.25),
        (16, 24, 3, 2, 6, 0.25),
        (24, 40, 5, 2, 6, 0.25),
        (40, 80, 3, 2, 6, 0.25),
        (80, 112, 5, 1, 6, 0.25),
        (112, 192, 5, 2, 6, 0.25),
        (192, 320, 3, 1, 6, 0.25)
    ]

    layers = []
    in_channels = 32

    # Initial convolution
    layers.append(nn.Sequential(
        nn.Conv2d(3, in_channels, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(in_channels),
        Swish()
    ))

    # MBConv blocks
    for out_channels, kernel_size, stride, expand_ratio, se_ratio in config:
        layers.append(MBConvBlock(in_channels, out_channels, kernel_size,
                                stride, expand_ratio, se_ratio))
        in_channels = out_channels

    # Final layers
    layers.append(nn.Sequential(
        nn.Conv2d(in_channels, 1280, kernel_size=1, bias=False),
        nn.BatchNorm2d(1280),
        Swish(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(0.2),
        nn.Linear(1280, num_classes)
    ))

    return nn.Sequential(*layers)
```

## 2. Vision Transformers

### 2.1 Vision Transformer (ViT) Architecture

#### 2.1.1 Mathematical Foundation
**Self-Attention Mechanism:**
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where:
- $Q$: Query matrix
- $K$: Key matrix
- $V$: Value matrix
- $d_k$: Dimension of keys

**Multi-Head Attention:**
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

#### 2.1.2 ViT Implementation

```python
class PatchEmbedding(nn.Module):
    """Convert image to patch embeddings"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super(PatchEmbedding, self).__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size,
                                   stride=patch_size)

    def forward(self, x):
        x = self.projection(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)  # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, embed_dim)
        return x

class PositionalEncoding(nn.Module):
    """Learnable positional encoding"""
    def __init__(self, n_patches, embed_dim):
        super(PositionalEncoding, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, n_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, x):
        B, n_patches, embed_dim = x.shape

        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional encoding
        x = x + self.pos_embedding[:, :(n_patches + 1)]

        return x

class TransformerEncoder(nn.Module):
    """Transformer encoder block"""
    def __init__(self, embed_dim, num_heads, mlp_ratio=4, dropout=0.1):
        super(TransformerEncoder, self).__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * mlp_ratio, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Self-attention with residual connection
        residual = x
        x = self.norm1(x)
        attn_output, _ = self.attention(x, x, x)
        x = residual + self.dropout1(attn_output)

        # MLP with residual connection
        residual = x
        x = self.norm2(x)
        x = residual + self.mlp(x)

        return x

class VisionTransformer(nn.Module):
    """Vision Transformer implementation"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 num_classes=1000, embed_dim=768, depth=12, num_heads=12,
                 mlp_ratio=4, dropout=0.1):
        super(VisionTransformer, self).__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.pos_encoding = PositionalEncoding(
            (img_size // patch_size) ** 2, embed_dim
        )

        # Transformer encoder layers
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoder(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)

        # Add positional encoding and class token
        x = self.pos_encoding(x)

        # Transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)

        # Classification head
        x = self.norm(x)
        cls_token = x[:, 0]  # Extract class token
        x = self.head(cls_token)

        return x
```

### 2.2 Swin Transformer

```python
class WindowAttention(nn.Module):
    """Window-based multi-head self-attention"""
    def __init__(self, dim, window_size, num_heads):
        super(WindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )

        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = nn.functional.softmax(attn, dim=-1)
        else:
            attn = nn.functional.softmax(attn, dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)

        return x

class SwinTransformerBlock(nn.Module):
    """Swin Transformer block with shifted window attention"""
    def __init__(self, dim, num_heads, window_size=7, shift_size=3, mlp_ratio=4):
        super(SwinTransformerBlock, self).__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim)
        )

    def forward(self, x):
        H, W = self.H, self.W
        B, L, C = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Window partition
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # Window attention
        attn_windows = self.attn(x_windows)

        # Window reverse
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x
```

## 3. Object Detection Architectures

### 3.1 Two-Stage Detectors (Faster R-CNN)

```python
class RegionProposalNetwork(nn.Module):
    """Region Proposal Network for Faster R-CNN"""
    def __init__(self, in_channels, mid_channels=512, n_anchors=9):
        super(RegionProposalNetwork, self).__init__()

        self.conv = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)

        # Classification layer (object vs background)
        self.cls_logits = nn.Conv2d(mid_channels, n_anchors * 2, kernel_size=1)

        # Regression layer (box coordinates)
        self.bbox_pred = nn.Conv2d(mid_channels, n_anchors * 4, kernel_size=1)

        self.init_weights()

    def init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, features):
        # Shared convolution
        x = F.relu(self.conv(features))

        # Classification and regression
        cls_logits = self.cls_logits(x)
        bbox_pred = self.bbox_pred(x)

        return cls_logits, bbox_pred

class ROIAlign(nn.Module):
    """Region of Interest Align pooling"""
    def __init__(self, output_size, spatial_scale):
        super(ROIAlign, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale

    def forward(self, features, rois):
        # Scale ROI coordinates
        rois = rois * self.spatial_scale

        # Perform ROI align pooling
        # Implementation would use bilinear interpolation
        pooled_features = torch.ops.torchvision.roi_align(
            features, rois, self.output_size, self.spatial_scale, 1
        )

        return pooled_features

class FasterRCNN(nn.Module):
    """Faster R-CNN implementation"""
    def __init__(self, backbone, num_classes, n_anchors=9):
        super(FasterRCNN, self).__init__()

        self.backbone = backbone
        self.num_classes = num_classes

        # Region Proposal Network
        self.rpn = RegionProposalNetwork(backbone.out_channels, n_anchors=n_anchors)

        # ROI Pooling
        self.roi_pool = ROIAlign((7, 7), spatial_scale=1/16)

        # Detection head
        self.detector_head = nn.Sequential(
            nn.Linear(256 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # Classification and regression heads
        self.cls_score = nn.Linear(1024, num_classes)
        self.bbox_pred = nn.Linear(1024, num_classes * 4)

    def forward(self, images, targets=None):
        # Extract features
        features = self.backbone(images)

        # Region proposals
        rpn_cls_logits, rpn_bbox_pred = self.rpn(features)

        # Generate proposals (simplified)
        proposals = self.generate_proposals(rpn_cls_logits, rpn_bbox_pred)

        if targets is not None:
            # Training mode
            roi_features = self.roi_pool(features, targets['boxes'])
            roi_features = roi_features.view(roi_features.size(0), -1)

            detector_features = self.detector_head(roi_features)

            cls_score = self.cls_score(detector_features)
            bbox_pred = self.bbox_pred(detector_features)

            return {
                'rpn_cls_loss': self.rpn_cls_loss(rpn_cls_logits, targets),
                'rpn_bbox_loss': self.rpn_bbox_loss(rpn_bbox_pred, targets),
                'detector_cls_loss': self.detector_cls_loss(cls_score, targets),
                'detector_bbox_loss': self.detector_bbox_loss(bbox_pred, targets)
            }
        else:
            # Inference mode
            roi_features = self.roi_pool(features, proposals)
            roi_features = roi_features.view(roi_features.size(0), -1)

            detector_features = self.detector_head(roi_features)

            cls_score = self.cls_score(detector_features)
            bbox_pred = self.bbox_pred(detector_features)

            return {
                'boxes': proposals,
                'scores': F.softmax(cls_score, dim=-1),
                'bbox_pred': bbox_pred
            }
```

### 3.2 Single-Stage Detectors (YOLO)

```python
class YOLOHead(nn.Module):
    """YOLO detection head"""
    def __init__(self, in_channels, num_classes, num_anchors=3):
        super(YOLOHead, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # Convolutional layers for detection
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels // 2, num_anchors * (5 + num_classes),
                     kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        # Apply detection head
        x = self.conv_layers(x)

        # Reshape output
        B, C, H, W = x.shape
        x = x.view(B, self.num_anchors, 5 + self.num_classes, H, W)
        x = x.permute(0, 1, 3, 4, 2)  # (B, anchors, H, W, 5 + num_classes)

        return x

class YOLOv5(nn.Module):
    """YOLOv5 implementation"""
    def __init__(self, backbone, num_classes=80):
        super(YOLOv5, self).__init__()

        self.backbone = backbone
        self.num_classes = num_classes

        # Detection heads for different scales
        self.head_small = YOLOHead(256, num_classes)
        self.head_medium = YOLOHead(512, num_classes)
        self.head_large = YOLOHead(1024, num_classes)

    def forward(self, x):
        # Extract features at different scales
        features = self.backbone(x)

        # Apply detection heads
        small_output = self.head_small(features['small'])
        medium_output = self.head_medium(features['medium'])
        large_output = self.head_large(features['large'])

        return {
            'small': small_output,
            'medium': medium_output,
            'large': large_output
        }
```

## 4. Segmentation Architectures

### 4.1 U-Net for Semantic Segmentation

```python
class DoubleConv(nn.Module):
    """Double convolution block for U-Net"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool and double conv"""
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    """U-Net architecture for semantic segmentation"""
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # Decoder
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        # Output layer
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Output
        logits = self.outc(x)

        return logits
```

### 4.2 DeepLabV3+ for Semantic Segmentation

```python
class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling"""
    def __init__(self, in_channels, out_channels, atrous_rates):
        super(ASPP, self).__init__()

        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ))

        for rate in atrous_rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate,
                         dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ))

        modules.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(modules) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))

        res[-1] = F.interpolate(res[-1], size=x.shape[2:], mode='bilinear', align_corners=False)

        res = torch.cat(res, dim=1)
        return self.project(res)

class DeepLabV3Plus(nn.Module):
    """DeepLabV3+ implementation"""
    def __init__(self, backbone, num_classes, atrous_rates=[6, 12, 18]):
        super(DeepLabV3Plus, self).__init__()

        self.backbone = backbone
        self.num_classes = num_classes

        # ASPP module
        self.aspp = ASPP(backbone.out_channels, 256, atrous_rates)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, x):
        input_size = x.shape[-2:]

        # Extract features
        features = self.backbone(x)
        low_level_features = features['low_level']
        high_level_features = features['high_level']

        # ASPP
        x = self.aspp(high_level_features)
        x = F.interpolate(x, size=low_level_features.shape[2:],
                         mode='bilinear', align_corners=False)

        # Decoder
        low_level_features = self.decoder(low_level_features)
        x = torch.cat([x, low_level_features], dim=1)
        x = self.final_conv(x)

        # Upsample to input size
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)

        return x
```

## 5. Self-Supervised Learning for Vision

### 5.1 Contrastive Learning (SimCLR)

```python
class ProjectionHead(nn.Module):
    """Projection head for contrastive learning"""
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(ProjectionHead, self).__init__()

        self.projection_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.projection_head(x)

class SimCLR(nn.Module):
    """SimCLR implementation for self-supervised learning"""
    def __init__(self, backbone, projection_dim=128, temperature=0.5):
        super(SimCLR, self).__init__()

        self.backbone = backbone
        self.projection_head = ProjectionHead(
            backbone.out_channels, 2048, projection_dim
        )
        self.temperature = temperature

    def forward(self, x1, x2):
        # Extract features
        features1 = self.backbone(x1)
        features2 = self.backbone(x2)

        # Project features
        projections1 = self.projection_head(features1)
        projections2 = self.projection_head(features2)

        # Normalize projections
        projections1 = F.normalize(projections1, dim=-1)
        projections2 = F.normalize(projections2, dim=-1)

        # Compute contrastive loss
        batch_size = projections1.shape[0]
        labels = torch.arange(batch_size).to(projections1.device)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(projections1, projections2.T) / self.temperature

        # Loss calculation
        loss = F.cross_entropy(similarity_matrix, labels)

        return loss
```

### 5.2 Masked Autoencoders (MAE)

```python
class MaskedAutoencoder(nn.Module):
    """Masked Autoencoder implementation"""
    def __init__(self, img_size=224, patch_size=16, embed_dim=768,
                 decoder_embed_dim=512, mask_ratio=0.75):
        super(MaskedAutoencoder, self).__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

        # Encoder
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, (img_size // patch_size) ** 2, embed_dim))

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8),
            num_layers=12
        )

        # Decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.decoder_pos_embed = nn.Parameter(torch.randn(1, (img_size // patch_size) ** 2, decoder_embed_dim))

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=decoder_embed_dim, nhead=8),
            num_layers=8
        )

        # Prediction head
        self.prediction_head = nn.Linear(decoder_embed_dim, patch_size * patch_size * 3)

    def forward(self, x, mask=None):
        # Patch embedding
        x = self.patch_embed(x)

        # Add positional embedding
        x = x + self.pos_embed

        # Apply mask
        if mask is None:
            mask = self.random_mask(x.shape[1])

        x_masked = x * mask.unsqueeze(-1)

        # Encode
        encoded = self.encoder(x_masked)

        # Decode
        decoder_input = self.decoder_embed(encoded)
        decoder_input = decoder_input + self.decoder_pos_embed

        decoded = self.decoder(decoder_input, decoder_input)

        # Predict
        predictions = self.prediction_head(decoded)

        return predictions, mask

    def random_mask(self, num_patches):
        """Generate random mask for patches"""
        mask = torch.ones(num_patches)
        num_masked = int(num_patches * self.mask_ratio)
        masked_indices = torch.randperm(num_patches)[:num_masked]
        mask[masked_indices] = 0
        return mask.to(self.pos_embed.device)
```

This comprehensive theoretical foundation covers the essential deep learning architectures and mathematical principles that form the basis of modern computer vision systems, from fundamental CNNs to advanced transformers and self-supervised learning approaches.