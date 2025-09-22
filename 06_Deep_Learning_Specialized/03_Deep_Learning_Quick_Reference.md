# Deep Learning Quick Reference Guide

## Essential Concepts and Architectures

---

## 1. Neural Network Fundamentals

### Activation Functions

| Function | Formula | Range | Use Case |
|----------|---------|-------|----------|
| **ReLU** | `f(x) = max(0, x)` | [0, ∞) | Hidden layers |
| **Leaky ReLU** | `f(x) = max(0.01x, x)` | (-∞, ∞) | Hidden layers |
| **Sigmoid** | `f(x) = 1/(1+e⁻ˣ)` | (0, 1) | Binary classification |
| **Tanh** | `f(x) = (eˣ - e⁻ˣ)/(eˣ + e⁻ˣ)` | (-1, 1) | Hidden layers |
| **Softmax** | `softmax(xᵢ) = eˣᵢ/∑ⱼeˣⱼ` | (0, 1) | Multi-class classification |
| **GELU** | `GELU(x) = x × Φ(x)` | (-∞, ∞) | Transformers |

**PyTorch Implementation:**
```python
import torch.nn as nn

# Common activation functions
activation = {
    'relu': nn.ReLU(),
    'leaky_relu': nn.LeakyReLU(0.01),
    'sigmoid': nn.Sigmoid(),
    'tanh': nn.Tanh(),
    'softmax': nn.Softmax(dim=1),
    'gelu': nn.GELU()
}
```

### Loss Functions

| Loss Function | Formula | Use Case |
|--------------|---------|----------|
| **MSE** | `MSE = (1/n)∑(yᵢ - ŷᵢ)²` | Regression |
| **MAE** | `MAE = (1/n)∑|yᵢ - ŷᵢ|` | Regression |
| **Cross-Entropy** | `CE = -∑yᵢlog(ŷᵢ)` | Classification |
| **Binary Cross-Entropy** | `BCE = -[ylog(ŷ) + (1-y)log(1-ŷ)]` | Binary classification |
| **Huber Loss** | `Lδ = {½(y-ŷ)² if \|y-ŷ\|≤δ, δ\|y-ŷ\|-½δ² otherwise}` | Robust regression |
| **Focal Loss** | `FL = -α(1-ŷ)ᵞlog(ŷ)` | Class imbalance |

### Optimization Algorithms

| Optimizer | Update Rule | Characteristics |
|-----------|-------------|------------------|
| **SGD** | `w = w - η∇w` | Simple, requires tuning |
| **SGD with Momentum** | `w = w - η∇w + βv` | Faster convergence |
| **Adam** | `m = β₁m + (1-β₁)∇w`<br>`v = β₂v + (1-β₂)(∇w)²`<br>`w = w - ηm/√v + ε` | Adaptive learning rate |
| **AdamW** | Adam + weight decay | Better generalization |
| **RMSprop** | `v = βv + (1-β)(∇w)²`<br>`w = w - η∇w/√v + ε` | Handles non-stationary objectives |

---

## 2. Deep Learning Architectures

### Convolutional Neural Networks (CNNs)

**Key Components:**
- **Convolutional Layer**: `output = input * kernel + bias`
- **Pooling Layer**: Reduces spatial dimensions
- **Batch Normalization**: Normalizes layer inputs
- **Residual Connection**: `output = F(x) + x`

**Common Architectures:**
```python
# Simple CNN
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(128 * 8 * 8, num_classes)

# ResNet Block
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = x
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        x += self.shortcut(residual)
        return F.relu(x)
```

### Recurrent Neural Networks (RNNs)

**LSTM Cell Equations:**
```
fₜ = σ(W_f·[hₜ₋₁, xₜ] + b_f)  # Forget gate
iₜ = σ(W_i·[hₜ₋₁, xₜ] + b_i)  # Input gate
oₜ = σ(W_o·[hₜ₋₁, xₜ] + b_o)  # Output gate
gₜ = tanh(W_g·[hₜ₋₁, xₜ] + b_g)  # Cell gate
Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ gₜ  # Cell state
hₜ = oₜ ⊙ tanh(Cₜ)  # Hidden state
```

**GRU Cell Equations:**
```
zₜ = σ(W_z·[hₜ₋₁, xₜ] + b_z)  # Update gate
rₜ = σ(W_r·[hₜ₋₁, xₜ] + b_r)  # Reset gate
nₜ = tanh(W_n·[rₜ ⊙ hₜ₋₁, xₜ] + b_n)  # New gate
hₜ = (1 - zₜ) ⊙ hₜ₋₁ + zₜ ⊙ nₜ  # Hidden state
```

### Transformers

**Self-Attention:**
```
Q = XW_Q, K = XW_K, V = XW_V
Attention(Q,K,V) = softmax(QKᵀ/√d_k)V
```

**Multi-Head Attention:**
```
MultiHead(Q,K,V) = Concat(head₁, ..., head_h)W_O
head_i = Attention(QW_Qᵢ, KW_Kᵢ, VW_Vᵢ)
```

**Positional Encoding:**
```
PE(pos,2i) = sin(pos/10000²ⁱ/d_model)
PE(pos,2i+1) = cos(pos/10000²ⁱ/d_model)
```

**Transformer Architecture:**
```python
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        # Self-attention
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))

        return x
```

---

## 3. Advanced Techniques

### Regularization Methods

| Method | Implementation | Effect |
|--------|----------------|---------|
| **L2 Regularization** | Add `λ∑w²` to loss | Prevents overfitting |
| **Dropout** | Randomly set neurons to 0 | Prevents co-adaptation |
| **BatchNorm** | Normalize layer inputs | Stabilizes training |
| **Early Stopping** | Stop when validation loss increases | Prevents overfitting |
| **Data Augmentation** | Apply transformations to training data | Increases effective dataset size |

```python
# Regularization in PyTorch
class RegularizedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# L2 regularization in optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
```

### Transfer Learning Strategies

| Method | Description | Use Case |
|--------|-------------|----------|
| **Feature Extraction** | Freeze backbone, train only head | Small datasets |
| **Fine-tuning** | Unfreeze some/all layers | Medium datasets |
| **Differential Learning Rates** | Different LR for backbone and head | Large datasets |
| **LoRA** | Low-rank adaptation of weights | Parameter-efficient fine-tuning |
| **Prompt Tuning** | Learn task-specific prompts | Large language models |

```python
# Transfer Learning Pipeline
class TransferLearningModel(nn.Module):
    def __init__(self, base_model, num_classes, freeze_backbone=True):
        super().__init__()
        self.base_model = base_model

        if freeze_backbone:
            for param in self.base_model.parameters():
                param.requires_grad = False

        # Replace classifier
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)
```

### Quantization Methods

| Type | Bits per Weight | Memory Reduction | Speedup |
|------|-----------------|------------------|---------|
| **FP32** | 32 | 1x | 1x |
| **FP16** | 16 | 2x | 2-3x |
| **INT8** | 8 | 4x | 4x |
| **INT4** | 4 | 8x | 8x |
| **Binary** | 1 | 32x | 32x |

```python
# Post-training quantization
def quantize_model(model):
    model.eval()
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)
    torch.quantization.convert(model, inplace=True)
    return model

# Quantization-aware training
def qat_training(model, train_loader):
    model.train()
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    torch.quantization.prepare_qat(model, inplace=True)

    for epoch in range(epochs):
        for batch in train_loader:
            # Normal training loop
            pass

    return torch.quantization.convert(model.eval())
```

### Distributed Training

| Method | Description | Requirements |
|--------|-------------|--------------|
| **Data Parallel** | Split data across GPUs | Multiple GPUs on single machine |
| **DistributedDataParallel** | More efficient data parallel | Multiple machines |
| **Model Parallel** | Split model across GPUs | Very large models |
| **Pipeline Parallel** | Parallelize model stages | Sequential layers |

```python
# Data Parallel
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

# Distributed Data Parallel
def setup_ddp(rank, world_size):
    torch.distributed.init_process_group(
        backend='nccl',
        rank=rank,
        world_size=world_size
    )
    model = nn.parallel.DistributedDataParallel(
        model.to(rank),
        device_ids=[rank]
    )
```

---

## 4. Framework-Specific Patterns

### PyTorch Best Practices

```python
# Training loop template
def train_model(model, train_loader, val_loader, epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(batch['input_ids'])
            loss = criterion(outputs, batch['labels'])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(batch['input_ids'])
                loss = criterion(outputs, batch['labels'])
                val_loss += loss.item()

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
```

### TensorFlow/Keras Patterns

```python
# Keras training template
def build_and_train_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=5),
        tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
    ]

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=100,
        callbacks=callbacks
    )

    return model, history
```

### JAX/Flax Patterns

```python
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax

class MLP(nn.Module):
    features: list

    def setup(self):
        self.layers = [nn.Dense(feat) for feat in self.features]

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = nn.relu(layer(x))
        return self.layers[-1](x)

def create_train_state(rng, learning_rate, input_shape, features):
    model = MLP(features)
    params = model.init(rng, jnp.ones(input_shape))['params']
    tx = optax.adamw(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )

@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch['x'])
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits, batch['y']
        ).mean()
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    return state.apply_gradients(grads=grads), loss
```

---

## 5. Performance Optimization

### Memory Optimization

| Technique | Memory Reduction | Implementation |
|-----------|-----------------|-----------------|
| **Gradient Checkpointing** | 2-5x | `torch.utils.checkpoint` |
| **Mixed Precision** | 2x | `torch.cuda.amp` |
| **Gradient Accumulation** | N/A | Accumulate gradients |
| **Model Pruning** | 2-10x | Remove unimportant weights |
| **Model Distillation** | Variable | Train smaller model |

```python
# Mixed precision training
def train_mixed_precision(model, train_loader):
    from torch.cuda.amp import GradScaler, autocast
    scaler = GradScaler()

    for batch in train_loader:
        optimizer.zero_grad()
        with autocast():
            outputs = model(batch['input_ids'])
            loss = criterion(outputs, batch['labels'])

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

### Inference Optimization

| Technique | Speedup | Implementation |
|-----------|---------|-----------------|
| **TorchScript** | 1.5-2x | `torch.jit.script` |
| **ONNX Export** | 1.5-3x | `torch.onnx.export` |
| **TensorRT** | 3-5x | NVIDIA TensorRT |
| **Pruning** | 1.2-2x | Remove channels/filters |
| **Quantization** | 2-4x | Reduce precision |

```python
# TorchScript optimization
def optimize_for_inference(model):
    model.eval()
    # Convert to TorchScript
    scripted_model = torch.jit.script(model)
    # Optimize
    scripted_model = torch.jit.optimize_for_inference(scripted_model)
    return scripted_model
```

---

## 6. Troubleshooting Common Issues

### Training Problems

| Issue | Possible Solutions |
|-------|-------------------|
| **Vanishing Gradients** | • Use ReLU/GELU activations<br>• Add batch normalization<br>• Use residual connections |
| **Exploding Gradients** | • Gradient clipping<br>• Lower learning rate<br>• Weight initialization |
| **Overfitting** | • Add regularization<br>• Increase dropout<br>• Data augmentation<br>• Early stopping |
| **Underfitting** | • Increase model complexity<br>• Reduce regularization<br>• Train longer |
| **Slow Training** | • Use mixed precision<br>• Increase batch size<br>• Use distributed training |
| **Memory Issues** | • Gradient checkpointing<br>• Reduce batch size<br>• Use gradient accumulation |

### Debugging Techniques

```python
# Gradient checking
def check_gradients(model, loss):
    model.zero_grad()
    loss.backward(retain_graph=True)

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm()
            param_norm = param.norm()
            print(f"{name}: grad_norm={grad_norm:.4f}, param_norm={param_norm:.4f}")

# Learning rate finder
def find_lr(model, train_loader, init_lr=1e-8, final_lr=10.0, beta=0.98):
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
    lr_mult = (final_lr / init_lr) ** (1 / len(train_loader))
    avg_loss = 0.0
    best_loss = float('inf')
    losses = []
    lrs = []

    for batch in train_loader:
        lr = init_lr * (lr_mult ** len(losses))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.zero_grad()
        outputs = model(batch['input_ids'])
        loss = criterion(outputs, batch['labels'])
        loss.backward()
        optimizer.step()

        avg_loss = beta * avg_loss + (1 - beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta ** (len(losses) + 1))

        if smoothed_loss < best_loss:
            best_loss = smoothed_loss

        if smoothed_loss > 4 * best_loss:
            break

        losses.append(smoothed_loss)
        lrs.append(lr)

    return lrs, losses
```

---

## 7. Latest Research Trends

### Efficient AI

| Trend | Description | Examples |
|-------|-------------|----------|
| **Mixture of Experts** | Sparse models with specialized sub-networks | Mixtral, GLaM |
| **Parameter-Efficient FT** | Fine-tune only a subset of parameters | LoRA, Prefix Tuning |
| **Sparse Training** | Train only important connections | RigL, SparseGPT |
| **Neural Architecture Search** | Automate architecture design | NAS-BERT, EfficientNet |
| **Knowledge Distillation** | Transfer knowledge from large to small models | DistilBERT, TinyBERT |

### Emerging Architectures

| Architecture | Key Innovation | Use Cases |
|--------------|----------------|-----------|
| **Mamba** | State space models | Long sequences |
| **RWKV** | RNN with Transformer benefits | Efficient LLMs |
| **Hyena** | Long convolution operators | Long-range dependencies |
| **Perceiver** | Cross-attention architecture | Multi-modal tasks |
| **GNN + Transformers** | Combine graph and sequence modeling | Molecular modeling |

---

## 8. Quick Reference Commands

### PyTorch Commands
```python
# Model operations
model.train()  # Set to training mode
model.eval()   # Set to evaluation mode
torch.save(model.state_dict(), 'model.pth')  # Save model
model.load_state_dict(torch.load('model.pth'))  # Load model

# GPU operations
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
torch.cuda.empty_cache()

# Tensor operations
x = torch.randn(10, 3, 32, 32)  # Random tensor
y = x.to(device)  # Move to GPU
z = x.cpu()  # Move to CPU
```

### Training Monitoring
```python
# Learning rate scheduling
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
scheduler.step()

# Early stopping
early_stopping = EarlyStopping(patience=10, min_delta=0.001)
if val_loss < best_val_loss:
    best_val_loss = val_loss
    torch.save(model.state_dict(), 'best_model.pth')
else:
    early_stopping(val_loss)
    if early_stopping.early_stop:
        break
```

This quick reference provides the essential deep learning concepts, architectures, and techniques that practitioners use daily. For detailed implementations and mathematical foundations, refer to the comprehensive guide.