# Comprehensive Deep Learning Terminology & Architecture Guide

## Table of Contents
1. [Neural Network Fundamentals](#neural-network-fundamentals)
2. [Deep Learning Architectures](#deep-learning-architectures)
3. [Advanced DL Techniques](#advanced-dl-techniques)
4. [DL Frameworks and Tools](#dl-frameworks-and-tools)

---

## Neural Network Fundamentals

### Perceptrons and Activation Functions

#### Perceptrons
The perceptron is the fundamental building block of neural networks. It's a linear binary classifier that makes predictions based on a weighted sum of inputs.

**Mathematical Formulation:**
```
y = f(∑ᵢ wᵢxᵢ + b)
```
where:
- `xᵢ` are input features
- `wᵢ` are weights
- `b` is bias
- `f` is the activation function

**PyTorch Implementation:**
```python
import torch
import torch.nn as nn

class Perceptron(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# Usage
perceptron = Perceptron(input_dim=10)
x = torch.randn(1, 10)
output = perceptron(x)
```

#### Activation Functions
Activation functions introduce non-linearity into neural networks, enabling them to learn complex patterns.

**Common Activation Functions:**

1. **ReLU (Rectified Linear Unit)**
   - Formula: `f(x) = max(0, x)`
   - Advantages: Computationally efficient, mitigates vanishing gradient
   - Disadvantages: Dying ReLU problem

```python
# ReLU and its variants
relu = nn.ReLU()
leaky_relu = nn.LeakyReLU(0.01)  # Allows small negative values
prelu = nn.PReLU(num_parameters=1)  # Learnable negative slope
elu = nn.ELU(alpha=1.0)  # Exponential linear unit
selu = nn.SELU()  # Scaled exponential linear unit
```

2. **Sigmoid**
   - Formula: `f(x) = 1 / (1 + e⁻ˣ)`
   - Range: (0, 1)
   - Use case: Binary classification output layer

```python
sigmoid = nn.Sigmoid()
```

3. **Softmax**
   - Formula: `softmax(xᵢ) = eˣᵢ / ∑ⱼ eˣⱼ`
   - Use case: Multi-class classification output layer

```python
softmax = nn.Softmax(dim=1)
log_softmax = nn.LogSoftmax(dim=1)  # For numerical stability
```

4. **GELU (Gaussian Error Linear Unit)**
   - Formula: `GELU(x) = x × Φ(x)` where Φ is the CDF of standard normal
   - Use case: Transformer architectures

```python
gelu = nn.GELU()
```

### Backpropagation and Gradient Descent

#### Backpropagation Algorithm
Backpropagation computes gradients efficiently using the chain rule of calculus.

**Key Steps:**
1. Forward pass: Compute predictions
2. Compute loss
3. Backward pass: Compute gradients using chain rule
4. Update weights using gradients

**Mathematical Foundation:**
For a network with layers L₁, L₂, ..., Lₙ:
```
∂Loss/∂w = (∂Loss/∂yₙ) × (∂yₙ/∂yₙ₋₁) × ... × (∂y₂/∂y₁) × (∂y₁/∂w)
```

**PyTorch Implementation:**
```python
# Manual backpropagation example
import torch

# Create tensors with requires_grad=True
x = torch.randn(10, requires_grad=True)
w = torch.randn(10, 1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

# Forward pass
y = torch.matmul(x.unsqueeze(0), w) + b
loss = torch.mean((y - 1.0)**2)

# Backward pass
loss.backward()

# Update weights
learning_rate = 0.01
with torch.no_grad():
    w -= learning_rate * w.grad
    b -= learning_rate * b.grad

    # Clear gradients
    w.grad.zero_()
    b.grad.zero_()
```

#### Gradient Descent Variants

1. **Stochastic Gradient Descent (SGD)**
```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

2. **Adam (Adaptive Moment Estimation)**
```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
```

3. **AdamW (Adam with Weight Decay)**
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

### Loss Functions

#### Regression Loss Functions

1. **Mean Squared Error (MSE)**
```python
mse_loss = nn.MSELoss()
```

2. **Mean Absolute Error (MAE)**
```python
mae_loss = nn.L1Loss()
```

3. **Huber Loss**
```python
huber_loss = nn.HuberLoss(delta=1.0)  # Combines MSE and MAE
```

#### Classification Loss Functions

1. **Binary Cross-Entropy**
```python
bce_loss = nn.BCELoss()
bce_logits_loss = nn.BCEWithLogitsLoss()  # More numerically stable
```

2. **Categorical Cross-Entropy**
```python
ce_loss = nn.CrossEntropyLoss()  # Includes LogSoftmax and NLLLoss
```

3. **Focal Loss (for class imbalance)**
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
```

### Regularization Techniques

#### Dropout
Dropout randomly sets neurons to zero during training to prevent overfitting.

```python
class DropoutNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.dropout = nn.Dropout(0.5)  # 50% dropout rate
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)
        return x
```

#### Batch Normalization
Batch normalization normalizes layer inputs to stabilize training.

```python
class BatchNormNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x
```

#### Layer Normalization
Layer normalization is commonly used in transformers.

```python
class LayerNormNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.ln1 = nn.LayerNorm(512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = torch.relu(self.ln1(self.fc1(x)))
        x = self.fc2(x)
        return x
```

#### Weight Decay (L2 Regularization)
```python
# Applied through optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
```

### Weight Initialization Methods

Proper initialization is crucial for training deep networks.

1. **Xavier/Glorot Initialization**
```python
# PyTorch default for linear layers
nn.init.xavier_uniform_(layer.weight)
nn.init.xavier_normal_(layer.weight)
```

2. **He Initialization**
```python
# Recommended for ReLU activations
nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
```

3. **Custom Initialization**
```python
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
```

### Learning Rate Schedules

Learning rate schedules adjust the learning rate during training.

1. **Step Decay**
```python
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
```

2. **Cosine Annealing**
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
```

3. **One Cycle Learning Rate**
```python
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.01,
    total_steps=len(train_loader) * epochs
)
```

4. **ReduceLROnPlateau**
```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.1,
    patience=10
)
```

---

## Deep Learning Architectures

### Convolutional Neural Networks (CNNs)

CNNs are specialized for processing grid-like data such as images.

#### Key Components

1. **Convolutional Layer**
```python
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x
```

2. **Pooling Layers**
```python
max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
```

3. **Complete CNN Architecture**
```python
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
```

#### Advanced CNN Architectures

1. **ResNet with Residual Connections**
```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = nn.functional.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
```

2. **VGG Network**
```python
class VGG(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Continue with more blocks...
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
```

### Recurrent Neural Networks (RNNs)

RNNs are designed for processing sequential data.

#### Basic RNN Implementation
```python
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size

        # Input to hidden
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        # Hidden to output
        self.h2o = nn.Linear(hidden_size, output_size)
        # Activation
        self.tanh = nn.Tanh()

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.tanh(self.i2h(combined))
        output = self.h2o(hidden)
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)
```

#### LSTM (Long Short-Term Memory)
LSTMs address the vanishing gradient problem in RNNs.

```python
class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # Input, forget, output, and cell gates
        self gates = nn.Linear(input_size + hidden_size, 4 * hidden_size)

    def forward(self, input, hidden):
        h_prev, c_prev = hidden

        # Concatenate input and previous hidden state
        combined = torch.cat((input, h_prev), 1)

        # Compute all gates
        gates = self.gates(combined)
        i, f, o, g = gates.chunk(4, 1)

        # Apply activations
        i = torch.sigmoid(i)  # Input gate
        f = torch.sigmoid(f)  # Forget gate
        o = torch.sigmoid(o)  # Output gate
        g = torch.tanh(g)     # Cell gate

        # Update cell state and hidden state
        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        # Forward pass
        out, _ = self.lstm(x, (h0, c0))
        return out
```

#### GRU (Gated Recurrent Unit)
GRU is a simplified version of LSTM.

```python
class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # Reset and update gates
        self.reset_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.update_gate = nn.Linear(input_size + hidden_size, hidden_size)
        # New gate
        self.new_gate = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)

        r = torch.sigmoid(self.reset_gate(combined))  # Reset gate
        z = torch.sigmoid(self.update_gate(combined))  # Update gate
        n = torch.tanh(self.new_gate(torch.cat((input, r * hidden), 1)))  # New gate

        hidden = (1 - z) * hidden + z * n
        return hidden

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.gru(x, h0)
        return out
```

### Transformers and Attention Mechanisms

Transformers have revolutionized NLP and are now used across various domains.

#### Self-Attention Mechanism
```python
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.size()

        # Linear projections and split into heads
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax
        attention_weights = torch.softmax(scores, dim=-1)

        # Apply attention to values
        output = torch.matmul(attention_weights, V)

        # Concatenate heads and apply final projection
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        output = self.out_proj(output)

        return output, attention_weights
```

#### Multi-Head Attention
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attention = SelfAttention(embed_dim, num_heads)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        residual = x
        x, attention_weights = self.attention(x, mask)
        x = self.dropout(x)
        x = self.layer_norm(x + residual)
        return x, attention_weights
```

#### Transformer Encoder Block
```python
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        # Self-attention
        x, attention_weights = self.attention(x, mask)

        # Feed-forward
        residual = x
        x = self.feed_forward(x)
        x = self.layer_norm(x + residual)

        return x, attention_weights
```

#### Complete Transformer Encoder
```python
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_len=512):
        super().__init__()
        self.embed_dim = embed_dim

        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_len, embed_dim)

        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        batch_size, seq_len = x.size()

        # Create position indices
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)

        # Embed tokens and positions
        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(positions)

        # Combine embeddings
        x = self.dropout(token_emb + pos_emb)

        # Pass through transformer blocks
        attention_weights = []
        for layer in self.layers:
            x, weights = layer(x, mask)
            attention_weights.append(weights)

        return x, attention_weights
```

### Autoencoders and VAEs

#### Vanilla Autoencoder
```python
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
```

#### Variational Autoencoder (VAE)
```python
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

        # Mean and variance
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_var = nn.Linear(hidden_dim // 2, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstructed = self.decode(z)
        return reconstructed, mu, log_var

    def loss_function(self, recon_x, x, mu, log_var):
        # Reconstruction loss
        BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')

        # KL divergence
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        return BCE + KLD
```

### Generative Adversarial Networks (GANs)

GANs consist of a generator and discriminator that compete against each other.

#### Basic GAN Implementation
```python
class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super().__init__()
        self.img_shape = img_shape

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.01),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.01),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.01),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, int(torch.prod(torch.tensor(img_shape)))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img

class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(int(torch.prod(torch.tensor(img_shape))), 512),
            nn.LeakyReLU(0.01),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

class GAN(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super().__init__()
        self.generator = Generator(latent_dim, img_shape)
        self.discriminator = Discriminator(img_shape)

        # Loss function
        self.criterion = nn.BCELoss()

        # Optimizers
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=0.0002)
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002)

    def train_step(self, real_imgs, device):
        batch_size = real_imgs.size(0)

        # Adversarial ground truths
        valid = torch.ones(batch_size, 1, device=device)
        fake = torch.zeros(batch_size, 1, device=device)

        # Train Generator
        self.optimizer_G.zero_grad()

        # Generate fake images
        z = torch.randn(batch_size, self.generator.latent_dim, device=device)
        gen_imgs = self.generator(z)

        # Generator loss
        g_loss = self.criterion(self.discriminator(gen_imgs), valid)
        g_loss.backward()
        self.optimizer_G.step()

        # Train Discriminator
        self.optimizer_D.zero_grad()

        # Loss for real images
        real_loss = self.criterion(self.discriminator(real_imgs), valid)

        # Loss for fake images
        fake_loss = self.criterion(self.discriminator(gen_imgs.detach()), fake)

        # Total discriminator loss
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        self.optimizer_D.step()

        return g_loss.item(), d_loss.item()
```

### Diffusion Models

Diffusion models generate data by gradually denoising random noise.

#### Denoising Diffusion Probabilistic Model (DDPM)
```python
class GaussianDiffusion(nn.Module):
    def __init__(self, model, timesteps=1000, sampling_timesteps=1000):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.sampling_timesteps = sampling_timesteps

        # Define beta schedule
        self.beta = self._beta_schedule(timesteps)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def _beta_schedule(self, timesteps, s=0.008):
        # Cosine schedule as in Improved DDPM
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)

    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion process"""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha_bar = torch.sqrt(self.alpha_bar[t])
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar[t])

        return sqrt_alpha_bar * x_start + sqrt_one_minus_alpha_bar * noise

    def p_losses(self, x_start, t, noise=None):
        """Compute loss for a given timestep"""
        if noise is None:
            noise = torch.randn_like(x_start)

        # Add noise to input
        x_noisy = self.q_sample(x_start, t, noise)

        # Predict noise
        predicted_noise = self.model(x_noisy, t)

        # MSE loss
        loss = nn.functional.mse_loss(predicted_noise, noise)
        return loss

    @torch.no_grad()
    def p_sample(self, x, t, t_index):
        """Reverse diffusion process"""
        betas_t = self.beta[t]
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - self.alpha_bar[t])
        sqrt_recip_alpha_t = torch.sqrt(1 / self.alpha[t])

        # Predict noise
        predicted_noise = self.model(x, t)

        # Compute mean
        model_mean = sqrt_recip_alpha_t * (x - betas_t * predicted_noise / sqrt_one_minus_alpha_bar_t)

        if t_index == 0:
            return model_mean
        else:
            # Add noise
            noise = torch.randn_like(x)
            variance = torch.sqrt(self.beta[t]) * noise
            return model_mean + variance

    @torch.no_grad()
    def sample(self, batch_size, shape):
        """Generate samples from noise"""
        img = torch.randn(batch_size, *shape)

        for i in reversed(range(0, self.timesteps)):
            img = self.p_sample(img, torch.full((batch_size,), i, dtype=torch.long), i)

        return img

# U-Net model for diffusion
class UNet(nn.Module):
    def __init__(self, channels=3, dim=64):
        super().__init__()

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim * 4)
        )

        # Encoder
        self.encoder = nn.ModuleList([
            self._make_layer(channels, dim),
            self._make_layer(dim, dim * 2),
            self._make_layer(dim * 2, dim * 4),
            self._make_layer(dim * 4, dim * 8),
        ])

        # Middle
        self.middle = self._make_layer(dim * 8, dim * 8)

        # Decoder
        self.decoder = nn.ModuleList([
            self._make_layer(dim * 16, dim * 4),
            self._make_layer(dim * 8, dim * 2),
            self._make_layer(dim * 4, dim),
            self._make_layer(dim * 2, dim),
        ])

        # Output
        self.output = nn.Conv2d(dim * 2, channels, 1)

    def _make_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU()
        )

    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_embed(t)

        # Encoder
        skips = []
        for layer in self.encoder:
            x = layer(x)
            skips.append(x)
            x = nn.functional.avg_pool2d(x, 2)

        # Middle
        x = self.middle(x)

        # Decoder
        for layer, skip in zip(self.decoder, reversed(skips)):
            x = nn.functional.interpolate(x, scale_factor=2)
            x = torch.cat([x, skip], dim=1)
            x = layer(x)

        return self.output(x)
```

### Neural Architecture Search (NAS)

Neural Architecture Search automates the design of neural network architectures.

#### Differentiable Architecture Search (DARTS)
```python
class MixedOp(nn.Module):
    def __init__(self, C, stride):
        super().__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))

class Cell(nn.Module):
    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction):
        super().__init__()
        self.reduction = reduction

        if reduction:
            op_names, indices = zip(*GENO['reduce'])
            concat = range(2 + steps - multiplier, 2 + steps)
        else:
            op_names, indices = zip(*GENO['normal'])
            concat = range(2 + steps - multiplier, 2 + steps)

        self._compile(C, op_names, indices, concat, steps)

    def _compile(self, C, op_names, indices, concat, steps):
        assert len(op_names) == len(indices)
        self._steps = steps
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True)
            self._ops.append(op)

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self.edges[2*i]]
            h2 = states[self.edges[2*i+1]]
            op1 = self._ops[2*i]
            op2 = self._ops[2*i+1]
            h1 = op1(h1)
            h2 = op2(h2)
            if self.training and drop_prob > 0.:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)
            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)

class Network(nn.Module):
    def __init__(self, C, num_classes, layers, auxiliary, genotype):
        super().__init__()
        self._layers = layers
        self._auxiliary = auxiliary

        stem_multiplier = 3
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHead(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        logits_aux = None
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux
```

---

## Advanced DL Techniques

### Transfer Learning and Fine-tuning

Transfer learning leverages pre-trained models for new tasks with limited data.

#### Transfer Learning Pipeline
```python
import torch
import torch.nn as nn
import torchvision.models as models

class TransferLearningModel(nn.Module):
    def __init__(self, num_classes, freeze_backbone=True):
        super().__init__()

        # Load pre-trained model
        self.backbone = models.resnet50(pretrained=True)

        # Freeze backbone layers
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace the final layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

# Training with differential learning rates
def train_with_differential_lr(model, train_loader, optimizer, criterion, device):
    model.train()

    # Set different learning rates for backbone and head
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

    optimizer = torch.optim.Adam([
        {'params': backbone_params, 'lr': 1e-5},
        {'params': head_params, 'lr': 1e-3}
    ])

    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
```

### LoRA (Low-Rank Adaptation)

LoRA enables parameter-efficient fine-tuning of large models.

```python
class LoRALayer(nn.Module):
    def __init__(self, original_layer, rank=8, alpha=16):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha

        # Freeze original layer
        for param in self.original_layer.parameters():
            param.requires_grad = False

        # LoRA matrices
        in_features = original_layer.in_features
        out_features = original_layer.out_features

        self.lora_A = nn.Parameter(torch.randn(rank, in_features))
        self.lora_B = nn.Parameter(torch.randn(out_features, rank))

        # Scaling factor
        self.scaling = alpha / rank

        # Initialize LoRA matrices
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        original_output = self.original_layer(x)
        lora_output = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return original_output + lora_output

class LoRAModel(nn.Module):
    def __init__(self, base_model, rank=8, alpha=16):
        super().__init__()
        self.base_model = base_model

        # Replace linear layers with LoRA layers
        for name, module in base_model.named_modules():
            if isinstance(module, nn.Linear):
                parent = base_model
                name_parts = name.split('.')

                for part in name_parts[:-1]:
                    parent = getattr(parent, part)

                setattr(parent, name_parts[-1], LoRALayer(module, rank, alpha))

    def forward(self, x):
        return self.base_model(x)
```

### QLoRA (Quantized LoRA)

QLoRA combines quantization with LoRA for even greater memory efficiency.

```python
import bitsandbytes as bnb

class QuantizedLinear(nn.Module):
    def __init__(self, in_features, out_features, quant_type='nf4'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quant_type = quant_type

        # Quantized weights
        if quant_type == 'nf4':
            self.weight = bnb.nn.Params4bit(torch.randn(out_features, in_features))
        else:
            self.weight = bnb.nn.Params4bit(torch.randn(out_features, in_features), requires_grad=False)

        # LoRA components
        self.rank = 8
        self.alpha = 16
        self.lora_A = nn.Parameter(torch.randn(self.rank, in_features))
        self.lora_B = nn.Parameter(torch.randn(out_features, self.rank))
        self.scaling = self.alpha / self.rank

        # Initialize LoRA
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # Dequantize weights for computation
        weight = self.weight.dequantize()

        # Original computation
        original_output = torch.nn.functional.linear(x, weight)

        # LoRA adaptation
        lora_output = torch.nn.functional.linear(x, self.lora_B @ self.lora_A) * self.scaling

        return original_output + lora_output
```

### Model Quantization

Quantization reduces model size and improves inference speed.

#### Post-Training Quantization
```python
def quantize_model(model, calibration_loader):
    # Prepare for quantization
    model.eval()
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

    # Fuse modules
    model_fused = torch.quantization.fuse_modules(model, [['conv', 'relu', 'bn']])

    # Prepare model
    model_prepared = torch.quantization.prepare(model_fused)

    # Calibrate with data
    with torch.no_grad():
        for data, _ in calibration_loader:
            model_prepared(data)

    # Quantize model
    quantized_model = torch.quantization.convert(model_prepared)

    return quantized_model
```

#### Quantization-Aware Training
```python
def qat_training(model, train_loader, epochs=10):
    # Set up QAT
    model.train()
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')

    # Fuse modules
    model_fused = torch.quantization.fuse_modules(model, [['conv', 'relu', 'bn']])

    # Prepare QAT
    model_prepared = torch.quantization.prepare_qat(model_fused)

    # Training loop
    optimizer = torch.optim.Adam(model_prepared.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model_prepared(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    # Convert to quantized model
    quantized_model = torch.quantization.convert(model_prepared.eval())

    return quantized_model
```

### Knowledge Distillation

Knowledge distillation trains a smaller "student" model to mimic a larger "teacher" model.

```python
class DistillationLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, target):
        # Hard loss (student vs ground truth)
        hard_loss = self.ce_loss(student_logits, target)

        # Soft loss (student vs teacher)
        soft_loss = self.kl_div(
            torch.nn.functional.log_softmax(student_logits / self.temperature, dim=1),
            torch.nn.functional.softmax(teacher_logits / self.temperature, dim=1)
        ) * (self.temperature ** 2)

        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss

class DistillationTrainer:
    def __init__(self, teacher_model, student_model, temperature=4.0, alpha=0.7):
        self.teacher = teacher_model
        self.student = student_model
        self.teacher.eval()  # Freeze teacher

        self.criterion = DistillationLoss(temperature, alpha)
        self.optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)

    def train_step(self, data, target):
        # Teacher prediction
        with torch.no_grad():
            teacher_logits = self.teacher(data)

        # Student prediction
        student_logits = self.student(data)

        # Compute loss
        loss = self.criterion(student_logits, teacher_logits, target)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
```

### Mixed Precision Training

Mixed precision training uses both FP16 and FP32 to speed up training.

```python
def train_mixed_precision(model, train_loader, optimizer, criterion, device):
    from torch.cuda.amp import GradScaler, autocast

    scaler = GradScaler()

    model.train()
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            # Mixed precision forward pass
            with autocast():
                output = model(data)
                loss = criterion(output, target)

            # Scale gradients
            scaler.scale(loss).backward()

            # Unscale gradients and clip
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update weights
            scaler.step(optimizer)
            scaler.update()
```

### Distributed Training

#### Data Parallelism
```python
def setup_data_parallel(model):
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    return model

def train_data_parallel(model, train_loader, optimizer, criterion, device):
    model.train()
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
```

#### DistributedDataParallel (DDP)
```python
import torch.distributed as dist
import torch.multiprocessing as mp

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initialize process group
    dist.init_process_group(
        backend='nccl',
        rank=rank,
        world_size=world_size
    )

def train_ddp(rank, world_size, model, train_loader, optimizer, criterion):
    # Setup DDP
    setup_ddp(rank, world_size)

    # Move model to device and wrap with DDP
    device = rank
    model = model.to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # Create distributed sampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_loader.dataset)

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

def main():
    world_size = torch.cuda.device_count()
    mp.spawn(train_ddp, args=(world_size,), nprocs=world_size, join=True)
```

---

## DL Frameworks and Tools

### PyTorch Advanced Features

#### PyTorch Lightning for Clean Training
```python
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

class LitModel(pl.LightningModule):
    def __init__(self, model, learning_rate=0.001):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

# Training with PyTorch Lightning
model = LitModel(your_model)
trainer = pl.Trainer(
    max_epochs=100,
    callbacks=[
        ModelCheckpoint(monitor='val_loss', save_top_k=1),
        EarlyStopping(monitor='val_loss', patience=10)
    ],
    gpus=1 if torch.cuda.is_available() else 0
)
trainer.fit(model, train_loader, val_loader)
```

#### Hugging Face Transformers
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

# Load pre-trained model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Dataset preparation
class Dataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train
trainer.train()
```

### TensorFlow/Keras Implementations

#### Keras Custom Layers
```python
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

class MultiHeadAttention(Layer):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query_dense = Dense(embed_dim)
        self.key_dense = Dense(embed_dim)
        self.value_dense = Dense(embed_dim)
        self.output_dense = Dense(embed_dim)

    def call(self, inputs, mask=None):
        batch_size = tf.shape(inputs)[0]

        # Linear projections
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        # Split into heads
        query = tf.reshape(query, [batch_size, -1, self.num_heads, self.head_dim])
        key = tf.reshape(key, [batch_size, -1, self.num_heads, self.head_dim])
        value = tf.reshape(value, [batch_size, -1, self.num_heads, self.head_dim])

        # Transpose for matrix multiplication
        query = tf.transpose(query, [0, 2, 1, 3])
        key = tf.transpose(key, [0, 2, 1, 3])
        value = tf.transpose(value, [0, 2, 1, 3])

        # Scaled dot-product attention
        scores = tf.matmul(query, key, transpose_b=True) / tf.sqrt(tf.cast(self.head_dim, tf.float32))

        if mask is not None:
            scores = tf.where(mask == 0, -1e9, scores)

        attention_weights = tf.nn.softmax(scores, axis=-1)
        output = tf.matmul(attention_weights, value)

        # Concatenate heads
        output = tf.transpose(output, [0, 2, 1, 3])
        output = tf.reshape(output, [batch_size, -1, self.embed_dim])

        # Final projection
        output = self.output_dense(output)

        return output

class TransformerEncoder(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='relu'),
            Dense(embed_dim)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def call(self, inputs, training=False, mask=None):
        # Self-attention
        attn_output = self.attention(inputs, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        # Feed-forward
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TransformerModel(Model):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, vocab_size, max_len=512):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_dim)
        self.pos_encoding = self.positional_encoding(max_len, embed_dim)

        self.encoder_layers = [TransformerEncoder(embed_dim, num_heads, ff_dim)
                              for _ in range(num_layers)]

        self.dropout = Dropout(0.1)
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling1D()
        self.classifier = Dense(2, activation='softmax')

    def positional_encoding(self, max_len, embed_dim):
        position = tf.range(max_len, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(tf.range(0, embed_dim, 2, dtype=tf.float32) *
                         (-tf.math.log(10000.0) / embed_dim))

        pe = tf.zeros((max_len, embed_dim))
        pe = pe.numpy()  # Convert to numpy for assignment

        pe[:, 0::2] = tf.sin(position * div_term).numpy()
        pe[:, 1::2] = tf.cos(position * div_term).numpy()

        return tf.convert_to_tensor(pe, dtype=tf.float32)

    def call(self, inputs, training=False, mask=None):
        seq_len = tf.shape(inputs)[1]

        # Embedding
        x = self.embedding(inputs)
        x = x + self.pos_encoding[:seq_len, :]
        x = self.dropout(x, training=training)

        # Encoder layers
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, training=training, mask=mask)

        # Classification
        x = self.global_avg_pool(x)
        x = self.classifier(x)

        return x

# Compile and train
model = TransformerModel(num_layers=4, embed_dim=128, num_heads=8,
                        ff_dim=512, vocab_size=10000)

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss=SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

### JAX for High-Performance Computing

JAX is increasingly popular for high-performance deep learning research.

```python
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state

class MLP(nn.Module):
    features: list

    def setup(self):
        self.layers = [nn.Dense(feat) for feat in self.features]

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = nn.relu(layer(x))
        return self.layers[-1](x)

# Training state
def create_train_state(rng, learning_rate, input_shape, features):
    model = MLP(features)
    params = model.init(rng, jnp.ones(input_shape))['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )

# Training step
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
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss

# Training loop
def train_model(state, train_loader, num_epochs):
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in train_loader:
            state, loss = train_step(state, batch)
            epoch_loss += loss
        print(f'Epoch {epoch}, Loss: {epoch_loss / len(train_loader)}')
    return state
```

### Accelerate Library for Distributed Training

Accelerate simplifies distributed training across various hardware configurations.

```python
from accelerate import Accelerator
from transformers import get_scheduler

def train_with_accelerate(model, train_loader, num_epochs=10):
    # Initialize accelerator
    accelerator = Accelerator()

    # Prepare model, optimizer, and data
    model, optimizer, train_loader = accelerator.prepare(
        model,
        torch.optim.Adam(model.parameters(), lr=1e-4),
        train_loader
    )

    # Learning rate scheduler
    num_training_steps = num_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        for batch in train_loader:
            with accelerator.accumulate(model):
                outputs = model(batch['input_ids'], labels=batch['labels'])
                loss = outputs.loss

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

    return model
```

### DeepSpeed Integration

DeepSpeed enables training of very large models with memory optimization.

```python
import deepspeed

def train_with_deepspeed(model, train_loader, config_path):
    # Load DeepSpeed configuration
    ds_config = json.load(open(config_path))

    # Initialize DeepSpeed engine
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config_params=ds_config
    )

    # Training loop
    model_engine.train()
    for epoch in range(num_epochs):
        for batch in train_loader:
            # Forward pass
            outputs = model_engine(batch['input_ids'], labels=batch['labels'])
            loss = outputs.loss

            # Backward pass
            model_engine.backward(loss)
            model_engine.step()

    return model_engine
```

### BitsAndBytes for Efficient Quantization

BitsAndBytes provides 8-bit and 4-bit quantization for memory efficiency.

```python
import bitsandbytes as bnb

class Linear8bit(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state = bnb.MatmulLtState()

    def forward(self, x):
        if self.state.CxB is not None:
            self.state.CxB = None

        # 8-bit matrix multiplication
        out = bnb.matmul(
            x.half(),
            self.weight.half(),
            state=self.state,
            threshold=self.state.threshold
        )

        if self.bias is not None:
            out = out + self.bias.half()

        return out

class QuantizedModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model

        # Replace linear layers with 8-bit versions
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                parent = self.model
                name_parts = name.split('.')

                for part in name_parts[:-1]:
                    parent = getattr(parent, part)

                setattr(parent, name_parts[-1], Linear8bit(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None
                ))

    def forward(self, x):
        return self.model(x)
```

### Performance Optimization Tips

1. **Memory Optimization**
```python
# Gradient checkpointing
from torch.utils.checkpoint import checkpoint

class CheckpointedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1000, 1000)
        self.layer2 = nn.Linear(1000, 1000)
        self.layer3 = nn.Linear(1000, 1000)

    def forward(self, x):
        x = checkpoint(self.layer1, x)
        x = checkpoint(self.layer2, x)
        x = self.layer3(x)
        return x
```

2. **CUDA Memory Management**
```python
def clear_cuda_cache():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def memory_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3  # GB
        print(f"Allocated: {allocated:.2f} GB, Cached: {cached:.2f} GB")
```

3. **Gradient Accumulation for Large Batch Sizes**
```python
def train_with_gradient_accumulation(model, train_loader, accumulation_steps=4):
    optimizer.zero_grad()

    for i, batch in enumerate(train_loader):
        outputs = model(batch['input_ids'], labels=batch['labels'])
        loss = outputs.loss / accumulation_steps
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
```

This comprehensive documentation covers the essential deep learning concepts, architectures, and techniques with practical implementations in PyTorch, TensorFlow, and JAX. Each section includes mathematical foundations, code examples, and best practices for production-level deep learning development.