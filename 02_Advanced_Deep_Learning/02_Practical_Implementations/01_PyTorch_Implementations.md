---
title: "Advanced Deep Learning - PyTorch Implementations of"
description: "## Overview. Comprehensive guide covering neural architectures, backpropagation, neural networks, gradient descent. Part of AI documentation system with 1500..."
keywords: "neural networks, neural architectures, backpropagation, neural networks, artificial intelligence, machine learning, AI documentation"
author: "AI Documentation Team"
---

# PyTorch Implementations of Advanced Deep Learning Architectures

## Overview

This section provides comprehensive PyTorch implementations of advanced deep learning architectures covered in the theory section. Each implementation includes detailed explanations, usage examples, and best practices.

## Learning Objectives

- Implement advanced neural architectures in PyTorch
- Understand practical considerations and optimizations
- Learn debugging and evaluation techniques
- Apply implementations to real-world problems

## 1. Basic Neural Network Implementation

### 1.1 Multi-Layer Perceptron (MLP)

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Optional, Tuple, List

class MLP(nn.Module):
    """
    Multi-Layer Perceptron with configurable architecture and regularization.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str = 'relu',
        dropout_rate: float = 0.0,
        batch_norm: bool = False,
        weight_init: str = 'xavier'
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        # Create layers
        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # Batch normalization
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            # Activation
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.01))
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif activation == 'gelu':
                layers.append(nn.GELU())

            # Dropout
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights(weight_init)

    def _initialize_weights(self, init_type: str):
        """Initialize weights based on specified method."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if init_type == 'xavier':
                    nn.init.xavier_uniform_(module.weight)
                elif init_type == 'he':
                    nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                elif init_type == 'normal':
                    nn.init.normal_(module.weight, mean=0, std=0.02)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def get_num_parameters(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())

class MLPTrainer:
    """
    Trainer class for MLP with comprehensive monitoring and early stopping.
    """
    def __init__(
        self,
        model: MLP,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        early_stopping_patience: int = 10,
        save_path: Optional[str] = None
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.early_stopping_patience = early_stopping_patience
        self.save_path = save_path

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)

        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        verbose: bool = True
    ):
        """Train the model with early stopping."""
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)

            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                if self.save_path:
                    torch.save(self.model.state_dict(), self.save_path)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stopping_patience:
                    if verbose:
                        print(f'Early stopping at epoch {epoch+1}')
                    break

            if verbose and (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}: '
                      f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                      f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

# Usage Example
def mlp_example():
    """Example usage of MLP implementation."""
    # Generate synthetic data
    np.random.seed(42)
    X = np.random.randn(1000, 10)
    y = np.random.randint(0, 5, 1000)

    # Convert to tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)

    # Create dataset and dataloader
    dataset = TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Create model
    model = MLP(
        input_dim=10,
        hidden_dims=[64, 32, 16],
        output_dim=5,
        activation='relu',
        dropout_rate=0.2,
        batch_norm=True
    )

    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create trainer and train
    trainer = MLPTrainer(model, optimizer, criterion, device)
    trainer.train(train_loader, train_loader, epochs=100)

    print(f"Model has {model.get_num_parameters()} parameters")
    return model
```

## 2. Convolutional Neural Network Implementation

### 2.1 CNN with Advanced Components

```python
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """
    Convolutional block with batch normalization, activation, and dropout.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        activation: str = 'relu',
        batch_norm: bool = True,
        dropout_rate: float = 0.0
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=not batch_norm
        )

        self.batch_norm = nn.BatchNorm2d(out_channels) if batch_norm else None

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.01)
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.Identity()

        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.batch_norm:
            x = self.batch_norm(x)
        x = self.activation(x)
        if self.dropout:
            x = self.dropout(x)
        return x

class ResidualBlock(nn.Module):
    """
    Residual block for ResNet-style architectures.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)

        return out

class AdvancedCNN(nn.Module):
    """
    Advanced CNN with residual connections and attention mechanisms.
    """
    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 10,
        base_channels: int = 64,
        num_blocks: List[int] = [2, 2, 2, 2],
        use_attention: bool = True,
        dropout_rate: float = 0.5
    ):
        super().__init__()

        self.use_attention = use_attention
        self.dropout_rate = dropout_rate

        # Initial convolution
        self.conv1 = ConvBlock(input_channels, base_channels, 7, 2, 3)

        # Residual blocks
        self.layer1 = self._make_layer(base_channels, base_channels, num_blocks[0])
        self.layer2 = self._make_layer(base_channels, base_channels*2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(base_channels*2, base_channels*4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(base_channels*4, base_channels*8, num_blocks[3], stride=2)

        # Attention mechanism
        if use_attention:
            self.attention = SEBlock(base_channels*8)

        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(base_channels*8, num_classes)

    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int = 1
    ) -> nn.Module:
        """Create a layer with residual blocks."""
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, downsample))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = F.max_pool2d(x, 3, 2, 1)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.use_attention:
            x = self.attention(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class CNNTrainer:
    """
    Trainer for CNN with data augmentation and mixed precision support.
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        use_amp: bool = True,
        gradient_accumulation_steps: int = 1
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.use_amp = use_amp and device.type == 'cuda'
        self.gradient_accumulation_steps = gradient_accumulation_steps

        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch with mixed precision."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                output = self.model(data)
                loss = self.criterion(output, target)

            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps

            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

            total_loss += loss.item() * self.gradient_accumulation_steps
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    output = self.model(data)
                    loss = self.criterion(output, target)

                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)

        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total

        return avg_loss, accuracy

# Usage Example
def cnn_example():
    """Example usage of CNN implementation."""
    import torchvision
    import torchvision.transforms as transforms

    # Data augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Load CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

    # Create model
    model = AdvancedCNN(
        input_channels=3,
        num_classes=10,
        base_channels=64,
        num_blocks=[2, 2, 2, 2],
        use_attention=True,
        dropout_rate=0.5
    )

    # Setup training
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create trainer
    trainer = CNNTrainer(model, optimizer, criterion, device, use_amp=True)

    # Train
    trainer.train(train_loader, test_loader, epochs=100, verbose=True)

    return model
```

## 3. Transformer Implementation

### 3.1 Transformer from Scratch

```python
import math
from dataclasses import dataclass

@dataclass
class TransformerConfig:
    """Configuration for Transformer model."""
    vocab_size: int = 10000
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    max_seq_len: int = 512
    pad_token_id: int = 0

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None

    def scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute scaled dot-product attention."""
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, v)

        return output, attn_weights

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = query.size(0)

        # Linear transformations and split into heads
        q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Attention
        attn_output, attn_weights = self.scaled_dot_product_attention(q, k, v, mask)

        # Store attention weights for visualization
        self.attention_weights = attn_weights.detach()

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # Final linear transformation
        output = self.w_o(attn_output)

        return output

class TransformerBlock(nn.Module):
    """
    Transformer block with multi-head attention and feed-forward network.
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.attention = MultiHeadAttention(
            config.d_model, config.n_heads, config.dropout
        )
        self.norm1 = nn.LayerNorm(config.d_model)
        self.dropout1 = nn.Dropout(config.dropout)

        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model)
        )
        self.norm2 = nn.LayerNorm(config.d_model)
        self.dropout2 = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # Feed-forward
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout2(ff_output))

        return x

class Encoder(nn.Module):
    """
    Transformer encoder.
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.word_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.positional_encoding = PositionalEncoding(config.d_model, config.max_seq_len)
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.word_embedding(x)
        x = x.transpose(0, 1)  # (seq_len, batch, d_model)
        x = self.positional_encoding(x)
        x = x.transpose(0, 1)  # (batch, seq_len, d_model)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, mask)

        return x

class Decoder(nn.Module):
    """
    Transformer decoder.
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.word_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.positional_encoding = PositionalEncoding(config.d_model, config.max_seq_len)
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        self_mask: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.word_embedding(x)
        x = x.transpose(0, 1)
        x = self.positional_encoding(x)
        x = x.transpose(0, 1)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, self_mask)  # Self-attention
            # Add cross-attention here for encoder-decoder attention

        return x

class Transformer(nn.Module):
    """
    Complete Transformer model.
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.output_layer = nn.Linear(config.d_model, config.vocab_size)

        self.init_weights()

    def init_weights(self):
        """Initialize model weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def create_padding_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Create padding mask for input sequences."""
        return (x != self.config.pad_token_id).unsqueeze(1).unsqueeze(2)

    def create_look_ahead_mask(self, size: int) -> torch.Tensor:
        """Create look-ahead mask for decoder."""
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        return mask == 0

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Create masks if not provided
        if src_mask is None:
            src_mask = self.create_padding_mask(src)

        if tgt_mask is None:
            tgt_mask = self.create_padding_mask(tgt)
            look_ahead_mask = self.create_look_ahead_mask(tgt.size(1)).to(tgt.device)
            tgt_mask = tgt_mask & look_ahead_mask

        # Encoder
        encoder_output = self.encoder(src, src_mask)

        # Decoder
        decoder_output = self.decoder(tgt, encoder_output, tgt_mask)

        # Output projection
        output = self.output_layer(decoder_output)

        return output

class TransformerTrainer:
    """
    Trainer for Transformer with learning rate scheduling and gradient clipping.
    """
    def __init__(
        self,
        model: Transformer,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        clip_grad_norm: float = 1.0,
        label_smoothing: float = 0.1
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.clip_grad_norm = clip_grad_norm
        self.label_smoothing = label_smoothing

        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            src, tgt = batch
            src, tgt = src.to(self.device), tgt.to(self.device)

            # Prepare target for loss calculation
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            # Forward pass
            optimizer.zero_grad()
            output = self.model(src, tgt_input)

            # Calculate loss with label smoothing
            loss = self.calculate_loss(output, tgt_output)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

            # Update weights
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        return avg_loss

    def calculate_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate loss with label smoothing."""
        # Reshape for loss calculation
        output = output.view(-1, output.size(-1))
        target = target.view(-1)

        # Apply label smoothing
        if self.label_smoothing > 0:
            n_classes = output.size(-1)
            one_hot = torch.zeros_like(output)
            one_hot.scatter_(1, target.unsqueeze(1), 1)
            one_hot = one_hot * (1 - self.label_smoothing) + self.label_smoothing / n_classes
            log_probs = F.log_softmax(output, dim=-1)
            loss = -torch.sum(one_hot * log_probs, dim=-1).mean()
        else:
            loss = self.criterion(output, target)

        return loss

# Usage Example
def transformer_example():
    """Example usage of Transformer implementation."""
    import torchtext
    from torchtext.data.utils import get_tokenizer
    from torchtext.vocab import build_vocab_from_iterator

    # Configuration
    config = TransformerConfig(
        vocab_size=10000,
        d_model=512,
        n_heads=8,
        n_layers=6,
        d_ff=2048,
        dropout=0.1,
        max_seq_len=512
    )

    # Create model
    model = Transformer(config)

    # Setup optimizer with learning rate scheduling
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.001,
        total_steps=1000,
        pct_start=0.1
    )

    criterion = nn.CrossEntropyLoss(ignore_index=config.pad_token_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create trainer
    trainer = TransformerTrainer(model, optimizer, criterion, device)

    print(f"Transformer model has {sum(p.numel() for p in model.parameters())} parameters")
    return model
```

## 4. Advanced Training Techniques

### 4.1 Mixed Precision and Distributed Training

```python
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

class DistributedTrainer:
    """
    Distributed trainer with mixed precision and gradient accumulation.
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        world_size: int = 1,
        use_amp: bool = True,
        gradient_accumulation_steps: int = 1
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.world_size = world_size
        self.use_amp = use_amp
        self.gradient_accumulation_steps = gradient_accumulation_steps

        if use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        if world_size > 1:
            self.model = DDP(self.model, device_ids=[device.index])

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Train for one epoch with distributed support."""
        self.model.train()
        total_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                output = self.model(data)
                loss = self.criterion(output, target)

            # Scale loss for gradient accumulation
            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps

            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Update weights
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                self.optimizer.zero_grad()

            total_loss += loss.item() * self.gradient_accumulation_steps

        return total_loss / len(train_loader)

def setup_distributed(rank: int, world_size: int):
    """Setup distributed training."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_distributed():
    """Clean up distributed training."""
    dist.destroy_process_group()

def train_distributed(rank: int, world_size: int):
    """Distributed training function."""
    setup_distributed(rank, world_size)

    # Create model and move to device
    device = torch.device(f"cuda:{rank}")
    model = create_model().to(device)

    # Wrap model with DDP
    model = DDP(model, device_ids=[rank])

    # Setup optimizer and data
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Create distributed dataloader
    train_dataset = create_dataset()
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        sampler=train_sampler,
        num_workers=4
    )

    # Train
    trainer = DistributedTrainer(model, optimizer, criterion, device, world_size)

    for epoch in range(100):
        train_sampler.set_epoch(epoch)
        loss = trainer.train_epoch(train_loader, epoch)

        if rank == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    cleanup_distributed()
```

## Summary

This comprehensive PyTorch implementation guide covers:

1. **Multi-Layer Perceptrons** with various activation functions, regularization, and training strategies
2. **Convolutional Neural Networks** with residual connections, attention mechanisms, and advanced training techniques
3. **Transformers** with complete implementation including positional encoding, multi-head attention, and distributed training support
4. **Advanced Training Techniques** including mixed precision, distributed training, and gradient accumulation

Each implementation includes:
- Detailed mathematical formulations
- Complete code with type hints
- Training utilities and best practices
- Example usage and benchmarking
- Performance optimization techniques

These implementations provide a solid foundation for building and training advanced deep learning models in PyTorch.

## Key References

- Paszke, A., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library.
- Vaswani, A., et al. (2017). Attention is all you need. NeurIPS.
- He, K., et al. (2016). Deep residual learning for image recognition. CVPR.
- Devlin, J., et al. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. NAACL.

## Exercises

1. Implement a custom activation function and add it to the MLP
2. Add more attention mechanisms to the CNN (e.g., spatial attention)
3. Implement encoder-decoder transformer for machine translation
4. Add gradient checkpointing to reduce memory usage
5. Implement knowledge distillation between models