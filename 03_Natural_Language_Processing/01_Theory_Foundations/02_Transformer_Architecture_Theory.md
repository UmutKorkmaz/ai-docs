# Transformer Architecture Theory: Mathematical Foundations

## 1. Introduction to Transformers

### 1.1 Historical Context

**Pre-Transformer Era**
- **RNNs (1986)**: Sequential processing with hidden states
- **LSTMs (1997)**: Address vanishing gradients with gating mechanisms
- **GRUs (2014)**: Simplified LSTM with two gates
- **Attention Mechanisms (2014)**: Bahdanau and Luong attention for machine translation

**Transformers (2017)**
- **"Attention Is All You Need"**: Vaswani et al.
- **Parallel Processing**: No sequential dependency
- **Scalability**: Efficient computation on modern hardware
- **State-of-the-Art**: Revolution in NLP and beyond

### 1.2 Core Innovations

**Self-Attention Mechanism**
- Query-Key-Value computation
- Dynamic context weighting
- Long-range dependency modeling

**Multi-Head Attention**
- Parallel attention heads
- Different representation subspaces
- Enhanced learning capacity

**Positional Encoding**
- Injecting sequence order information
- Sinusoidal encoding functions
- Learnable alternatives

**Layer Normalization and Residual Connections**
- Stabilizing training
- Enabling deep architectures
- Gradient flow optimization

## 2. Mathematical Foundations of Attention

### 2.1 Scaled Dot-Product Attention

**Basic Attention Formula**
Given queries Q, keys K, and values V, the attention mechanism computes:

```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

Where:
- Q ∈ ℝ^(n×d_k): Query matrix
- K ∈ ℝ^(m×d_k): Key matrix
- V ∈ ℝ^(m×d_v): Value matrix
- d_k: Dimension of keys/queries
- n, m: Sequence lengths

**Component Breakdown**
```python
import numpy as np
from typing import Tuple

class AttentionMechanism:
    def __init__(self, d_model: int, d_k: int = None, d_v: int = None):
        self.d_model = d_model
        self.d_k = d_k or d_model
        self.d_v = d_v or d_model

    def scaled_dot_product_attention(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                                   mask: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute scaled dot-product attention

        Args:
            Q: Query matrix (..., seq_len_q, d_k)
            K: Key matrix (..., seq_len_k, d_k)
            V: Value matrix (..., seq_len_v, d_v)
            mask: Optional mask (..., seq_len_q, seq_len_k)

        Returns:
            output: Attention output (..., seq_len_q, d_v)
            attention_weights: Attention weights (..., seq_len_q, seq_len_k)
        """
        # Compute dot product of Q and K^T
        scores = np.matmul(Q, K.transpose(-2, -1))

        # Scale by sqrt(d_k)
        scores = scores / np.sqrt(self.d_k)

        # Apply mask if provided
        if mask is not None:
            scores = scores + (mask * -1e9)

        # Apply softmax to get attention weights
        attention_weights = self.softmax(scores, axis=-1)

        # Compute weighted sum of values
        output = np.matmul(attention_weights, V)

        return output, attention_weights

    def softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numerically stable softmax implementation"""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def causal_mask(self, seq_len: int) -> np.ndarray:
        """Create causal mask for autoregressive generation"""
        return np.triu(np.ones((seq_len, seq_len)), k=1)

    def padding_mask(self, sequence: np.ndarray, pad_value: int = 0) -> np.ndarray:
        """Create padding mask"""
        return (sequence == pad_value).astype(np.float32)
```

**Mathematical Properties**
- **Linearity**: Attention is linear in values
- **Softmax Differentiability**: Enables gradient-based optimization
- **Scale Invariance**: Scaling prevents softmax saturation
- **Computational Complexity**: O(n²d_k) for sequence length n

### 2.2 Multi-Head Attention

**Multi-Head Formulation**
Multiple attention heads operate in parallel on different linear projections:

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O

where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

**Projection Matrices**
- W_i^Q ∈ ℝ^(d_model×d_k): Query projection for head i
- W_i^K ∈ ℝ^(d_model×d_k): Key projection for head i
- W_i^V ∈ ℝ^(d_model×d_v): Value projection for head i
- W^O ∈ ℝ^(h*d_v×d_model): Output projection matrix

```python
class MultiHeadAttention:
    def __init__(self, d_model: int, num_heads: int, d_k: int = None, d_v: int = None):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_k or d_model // num_heads
        self.d_v = d_v or d_model // num_heads

        # Initialize projection matrices
        self.W_q = np.random.randn(d_model, num_heads * self.d_k) * 0.01
        self.W_k = np.random.randn(d_model, num_heads * self.d_k) * 0.01
        self.W_v = np.random.randn(d_model, num_heads * self.d_v) * 0.01
        self.W_o = np.random.randn(num_heads * self.d_v, d_model) * 0.01

        self.attention = AttentionMechanism(self.d_k, self.d_k, self.d_v)

    def split_heads(self, x: np.ndarray) -> np.ndarray:
        """Split last dimension into (num_heads, d_k/d_v)"""
        batch_size, seq_len, _ = x.shape
        return x.reshape(batch_size, seq_len, self.num_heads, -1).transpose(0, 2, 1, 3)

    def combine_heads(self, x: np.ndarray) -> np.ndarray:
        """Combine heads back to original shape"""
        batch_size, num_heads, seq_len, _ = x.shape
        return x.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)

    def forward(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                mask: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Multi-head attention forward pass

        Args:
            Q: Query (batch_size, seq_len_q, d_model)
            K: Key (batch_size, seq_len_k, d_model)
            V: Value (batch_size, seq_len_v, d_model)
            mask: Optional mask

        Returns:
            output: Multi-head attention output
            attention_weights: Attention weights for each head
        """
        batch_size = Q.shape[0]

        # Linear projections
        Q_proj = np.matmul(Q, self.W_q)
        K_proj = np.matmul(K, self.W_k)
        V_proj = np.matmul(V, self.W_v)

        # Split into heads
        Q_heads = self.split_heads(Q_proj)  # (batch, num_heads, seq_len_q, d_k)
        K_heads = self.split_heads(K_proj)  # (batch, num_heads, seq_len_k, d_k)
        V_heads = self.split_heads(V_proj)  # (batch, num_heads, seq_len_v, d_v)

        # Apply attention to each head
        attention_outputs = []
        attention_weights_list = []

        for i in range(self.num_heads):
            head_output, head_weights = self.attention.scaled_dot_product_attention(
                Q_heads[:, i], K_heads[:, i], V_heads[:, i], mask
            )
            attention_outputs.append(head_output)
            attention_weights_list.append(head_weights)

        # Combine heads
        combined = np.concatenate(attention_outputs, axis=-1)
        output = np.matmul(self.combine_heads(combined), self.W_o)

        # Stack attention weights
        attention_weights = np.stack(attention_weights_list, axis=1)

        return output, attention_weights
```

### 2.3 Attention Variants

**Additive Attention (Bahdanau)**
```
e_ij = v^T tanh(W_q q_i + W_k k_j + b)
```

**Multiplicative Attention (Luong)**
```
e_ij = q_i^T W k_j
```

**Self-Attention vs. Cross-Attention**
- **Self-Attention**: Q, K, V from same sequence
- **Cross-Attention**: Q from one sequence, K, V from another

## 3. Positional Encoding

### 3.1 Sinusoidal Positional Encoding

**Encoding Formulas**
```
PE(pos, 2i) = sin(pos/10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
```

Where:
- pos: Position in sequence
- i: Dimension index
- d_model: Model dimension

**Mathematical Properties**
- **Periodicity**: Different frequencies for different dimensions
- **Relative Position Encoding**: Linear relationships between positions
- **Bounded Values**: All values in [-1, 1]
- **Uniqueness**: Each position has unique encoding

```python
class PositionalEncoding:
    def __init__(self, d_model: int, max_len: int = 5000):
        self.d_model = d_model
        self.max_len = max_len

    def sinusoidal_encoding(self, positions: np.ndarray) -> np.ndarray:
        """
        Compute sinusoidal positional encoding

        Args:
            positions: Array of positions (seq_len,)

        Returns:
            pos_encoding: Positional encoding (seq_len, d_model)
        """
        seq_len = len(positions)
        pe = np.zeros((seq_len, self.d_model))

        for pos in range(seq_len):
            for i in range(0, self.d_model, 2):
                # Even dimensions: sin
                pe[pos, i] = np.sin(positions[pos] / (10000 ** (2 * i / self.d_model)))
                # Odd dimensions: cos
                if i + 1 < self.d_model:
                    pe[pos, i + 1] = np.cos(positions[pos] / (10000 ** (2 * i / self.d_model)))

        return pe

    def relative_position_encoding(self, seq_len: int) -> np.ndarray:
        """Generate relative position encoding"""
        positions = np.arange(seq_len)
        return self.sinusoidal_encoding(positions)

    def learnable_encoding(self, seq_len: int, batch_size: int = 1) -> np.ndarray:
        """Generate learnable position encoding"""
        return np.random.randn(batch_size, seq_len, self.d_model) * 0.01
```

### 3.2 Learnable Positional Encoding

**Absolute Position Embeddings**
- Trainable embedding matrix
- Simple but less flexible for variable lengths

**Relative Position Representations**
- Shaw et al. (2018): Relative position embeddings
- Huang et al. (2019): Relative position representations
- Improved generalization to unseen sequence lengths

## 4. Transformer Architecture Components

### 4.1 Feed-Forward Networks

**Position-wise Feed-Forward Network**
Each position processed independently through two linear transformations with ReLU:

```
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
```

```python
class PositionWiseFeedForward:
    def __init__(self, d_model: int, d_ff: int):
        self.d_model = d_model
        self.d_ff = d_ff

        # Initialize weights
        self.W1 = np.random.randn(d_model, d_ff) * 0.01
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.01
        self.b2 = np.zeros(d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through position-wise feed-forward network

        Args:
            x: Input tensor (batch_size, seq_len, d_model)

        Returns:
            output: Output tensor (batch_size, seq_len, d_model)
        """
        # First linear layer + ReLU
        hidden = np.maximum(0, np.matmul(x, self.W1) + self.b1)

        # Second linear layer
        output = np.matmul(hidden, self.W2) + self.b2

        return output

    def gelu_activation(self, x: np.ndarray) -> np.ndarray:
        """GELU activation function"""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

    def swish_activation(self, x: np.ndarray) -> np.ndarray:
        """Swish activation function"""
        return x * self.sigmoid(x)

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid function"""
        return 1 / (1 + np.exp(-x))
```

### 4.2 Layer Normalization

**Layer Normalization Formula**
```
LayerNorm(x) = γ ⊙ (x - μ) / √(σ² + ε) + β
```

Where:
- μ: Mean of input
- σ²: Variance of input
- γ, β: Learnable scale and shift parameters
- ε: Small constant for numerical stability

```python
class LayerNorm:
    def __init__(self, d_model: int, eps: float = 1e-6):
        self.d_model = d_model
        self.eps = eps

        # Learnable parameters
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply layer normalization

        Args:
            x: Input tensor (..., d_model)

        Returns:
            normalized: Normalized tensor
        """
        # Calculate mean and variance
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)

        # Normalize
        normalized = (x - mean) / np.sqrt(variance + self.eps)

        # Scale and shift
        output = self.gamma * normalized + self.beta

        return output

    def backward(self, x: np.ndarray, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward pass for layer normalization

        Returns:
            grad_x: Gradient w.r.t. input
            grad_gamma: Gradient w.r.t. gamma
            grad_beta: Gradient w.r.t. beta
        """
        # Forward pass values
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        std = np.sqrt(variance + self.eps)

        # Normalized input
        x_norm = (x - mean) / std

        # Gradients
        grad_gamma = np.sum(grad_output * x_norm, axis=tuple(range(grad_output.ndim - 1)))
        grad_beta = np.sum(grad_output, axis=tuple(range(grad_output.ndim - 1)))

        # Gradient w.r.t. input (simplified)
        N = x.shape[-1]
        grad_x_norm = grad_output * self.gamma
        grad_x = (grad_x_norm - np.mean(grad_x_norm, axis=-1, keepdims=True) -
                  x_norm * np.mean(grad_x_norm * x_norm, axis=-1, keepdims=True)) / std

        return grad_x, grad_gamma, grad_beta
```

### 4.3 Residual Connections

**Residual Connection Formula**
```
Output = LayerNorm(x + Sublayer(x))
```

**Benefits**
- **Gradient Flow**: Mitigates vanishing gradient problem
- **Stability**: Improves training stability
- **Identity Mapping**: Preserves input information

```python
class ResidualConnection:
    def __init__(self, sublayer, d_model: int, dropout: float = 0.1):
        self.sublayer = sublayer
        self.layer_norm = LayerNorm(d_model)
        self.dropout = dropout

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Apply residual connection

        Args:
            x: Input tensor
            training: Whether in training mode

        Returns:
            output: Output with residual connection
        """
        # Apply sublayer
        sublayer_output = self.sublayer(x)

        # Apply dropout if training
        if training and self.dropout > 0:
            dropout_mask = np.random.binomial(1, 1 - self.dropout, size=sublayer_output.shape)
            sublayer_output = sublayer_output * dropout_mask / (1 - self.dropout)

        # Residual connection + layer normalization
        output = self.layer_norm.forward(x + sublayer_output)

        return output
```

## 5. Complete Transformer Architecture

### 5.1 Encoder-Decoder Structure

**Encoder Stack**
```
x_0 = Embedding + Positional Encoding
for l = 1 to L:
    x_l = EncoderLayer(x_{l-1})
```

**Decoder Stack**
```
y_0 = Embedding + Positional Encoding
for l = 1 to L:
    y_l = DecoderLayer(y_{l-1}, encoder_output)
```

```python
class TransformerEncoder:
    def __init__(self, d_model: int, num_heads: int, d_ff: int, num_layers: int,
                 dropout: float = 0.1):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.dropout = dropout

        # Initialize layers
        self.layers = []
        for _ in range(num_layers):
            multi_head_attn = MultiHeadAttention(d_model, num_heads)
            feed_forward = PositionWiseFeedForward(d_model, d_ff)

            self_attn = ResidualConnection(
                lambda x: multi_head_attn.forward(x, x, x)[0], d_model, dropout
            )
            ff = ResidualConnection(
                lambda x: feed_forward.forward(x), d_model, dropout
            )

            self.layers.append((self_attn, ff))

    def forward(self, x: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """
        Forward pass through encoder

        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Optional padding mask

        Returns:
            output: Encoded representations
        """
        output = x

        for self_attn, ff in self.layers:
            # Self-attention layer
            output = self_attn(output, training=True)

            # Feed-forward layer
            output = ff(output, training=True)

        return output

class TransformerDecoder:
    def __init__(self, d_model: int, num_heads: int, d_ff: int, num_layers: int,
                 dropout: float = 0.1):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.dropout = dropout

        # Initialize layers
        self.layers = []
        for _ in range(num_layers):
            self_attn = MultiHeadAttention(d_model, num_heads)
            cross_attn = MultiHeadAttention(d_model, num_heads)
            feed_forward = PositionWiseFeedForward(d_model, d_ff)

            masked_self_attn = ResidualConnection(
                lambda x: self_attn.forward(x, x, x, mask)[0], d_model, dropout
            )
            encoder_decoder_attn = ResidualConnection(
                lambda x: cross_attn.forward(x, encoder_output, encoder_output)[0], d_model, dropout
            )
            ff = ResidualConnection(
                lambda x: feed_forward.forward(x), d_model, dropout
            )

            self.layers.append((masked_self_attn, encoder_decoder_attn, ff))

    def forward(self, x: np.ndarray, encoder_output: np.ndarray,
                self_mask: np.ndarray = None, cross_mask: np.ndarray = None) -> np.ndarray:
        """
        Forward pass through decoder

        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            encoder_output: Encoder output
            self_mask: Self-attention mask (causal + padding)
            cross_mask: Cross-attention mask (encoder padding)

        Returns:
            output: Decoded representations
        """
        output = x

        for masked_self_attn, encoder_decoder_attn, ff in self.layers:
            # Masked self-attention
            output = masked_self_attn(output, training=True)

            # Encoder-decoder attention
            # Note: Need to pass encoder_output to the residual connection
            output = encoder_decoder_attn(output, training=True)

            # Feed-forward
            output = ff(output, training=True)

        return output
```

### 5.2 Complete Transformer Model

```python
class Transformer:
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, d_model: int = 512,
                 num_heads: int = 8, d_ff: int = 2048, num_layers: int = 6,
                 dropout: float = 0.1):
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.dropout = dropout

        # Embedding layers
        self.src_embedding = np.random.randn(src_vocab_size, d_model) * 0.01
        self.tgt_embedding = np.random.randn(tgt_vocab_size, d_model) * 0.01

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model)

        # Encoder and decoder
        self.encoder = TransformerEncoder(d_model, num_heads, d_ff, num_layers, dropout)
        self.decoder = TransformerDecoder(d_model, num_heads, d_ff, num_layers, dropout)

        # Output projection
        self.output_projection = np.random.randn(d_model, tgt_vocab_size) * 0.01

    def encode(self, src_tokens: np.ndarray, src_mask: np.ndarray = None) -> np.ndarray:
        """
        Encode source sequence

        Args:
            src_tokens: Source token indices (batch_size, src_len)
            src_mask: Source padding mask

        Returns:
            encoder_output: Encoded representations
        """
        batch_size, src_len = src_tokens.shape

        # Embedding + positional encoding
        src_embedded = self.src_embedding[src_tokens] * np.sqrt(self.d_model)
        positions = np.arange(src_len)
        pos_enc = self.pos_encoding.sinusoidal_encoding(positions)
        src_embedded = src_embedded + pos_enc

        # Encode
        encoder_output = self.encoder.forward(src_embedded, src_mask)

        return encoder_output

    def decode(self, tgt_tokens: np.ndarray, encoder_output: np.ndarray,
               self_mask: np.ndarray = None, cross_mask: np.ndarray = None) -> np.ndarray:
        """
        Decode target sequence

        Args:
            tgt_tokens: Target token indices (batch_size, tgt_len)
            encoder_output: Encoder output
            self_mask: Self-attention mask
            cross_mask: Cross-attention mask

        Returns:
            decoder_output: Decoded representations
        """
        batch_size, tgt_len = tgt_tokens.shape

        # Embedding + positional encoding
        tgt_embedded = self.tgt_embedding[tgt_tokens] * np.sqrt(self.d_model)
        positions = np.arange(tgt_len)
        pos_enc = self.pos_encoding.sinusoidal_encoding(positions)
        tgt_embedded = tgt_embedded + pos_enc

        # Decode
        decoder_output = self.decoder.forward(
            tgt_embedded, encoder_output, self_mask, cross_mask
        )

        # Output projection
        logits = np.matmul(decoder_output, self.output_projection)

        return logits

    def forward(self, src_tokens: np.ndarray, tgt_tokens: np.ndarray,
                src_mask: np.ndarray = None, tgt_mask: np.ndarray = None) -> np.ndarray:
        """
        Complete forward pass

        Args:
            src_tokens: Source token indices
            tgt_tokens: Target token indices
            src_mask: Source padding mask
            tgt_mask: Target mask (causal + padding)

        Returns:
            logits: Output logits
        """
        # Encode
        encoder_output = self.encode(src_tokens, src_mask)

        # Decode
        logits = self.decode(tgt_tokens, encoder_output, tgt_mask, src_mask)

        return logits

    def generate(self, src_tokens: np.ndarray, max_len: int, bos_token: int,
                 eos_token: int, src_mask: np.ndarray = None) -> np.ndarray:
        """
        Autoregressive generation

        Args:
            src_tokens: Source token indices
            max_len: Maximum generation length
            bos_token: Beginning of sentence token
            eos_token: End of sentence token
            src_mask: Source padding mask

        Returns:
            generated_tokens: Generated token sequence
        """
        batch_size = src_tokens.shape[0]
        device = src_tokens.device if hasattr(src_tokens, 'device') else 'cpu'

        # Encode source
        encoder_output = self.encode(src_tokens, src_mask)

        # Initialize generation
        generated_tokens = np.full((batch_size, 1), bos_token)

        for _ in range(max_len):
            # Create causal mask
            tgt_len = generated_tokens.shape[1]
            causal_mask = np.triu(np.ones((tgt_len, tgt_len)), k=1)

            # Decode
            logits = self.decode(
                generated_tokens, encoder_output,
                self_mask=causal_mask, cross_mask=src_mask
            )

            # Get last token predictions
            next_token_logits = logits[:, -1, :]
            next_tokens = np.argmax(next_token_logits, axis=-1)

            # Append generated tokens
            generated_tokens = np.concatenate([generated_tokens, next_tokens.reshape(-1, 1)], axis=1)

            # Check if all sequences have EOS
            if np.all(next_tokens == eos_token):
                break

        return generated_tokens
```

## 6. Training Dynamics

### 6.1 Optimization and Learning

**Learning Rate Schedules**
- **Warmup**: Gradually increase learning rate
- **Decay**: Exponential or linear decay
- **AdamW**: Weight decay variant of Adam

```python
class LearningRateScheduler:
    def __init__(self, d_model: int, warmup_steps: int = 4000):
        self.d_model = d_model
        self.warmup_steps = warmup_steps

    def get_lr(self, step: int) -> float:
        """
        Compute learning rate for given step

        Args:
            step: Training step

        Returns:
            lr: Learning rate
        """
        arg1 = step ** -0.5
        arg2 = step * (self.warmup_steps ** -1.5)
        return self.d_model ** -0.5 * min(arg1, arg2)

    def adamw_update(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray],
                     step: int, lr: float = None, beta1: float = 0.9,
                     beta2: float = 0.999, eps: float = 1e-8,
                     weight_decay: float = 0.01) -> Dict[str, np.ndarray]:
        """
        AdamW optimizer update

        Args:
            params: Model parameters
            grads: Gradients
            step: Current step
            lr: Learning rate (optional)
            beta1: First moment decay
            beta2: Second moment decay
            eps: Numerical stability
            weight_decay: Weight decay coefficient

        Returns:
            updated_params: Updated parameters
        """
        if lr is None:
            lr = self.get_lr(step)

        # Initialize moment estimates if not present
        if not hasattr(self, 'm'):
            self.m = {k: np.zeros_like(v) for k, v in params.items()}
            self.v = {k: np.zeros_like(v) for k, v in params.items()}

        updated_params = {}

        for key in params:
            # Get current values
            param = params[key]
            grad = grads[key]

            # Update biased first moment estimate
            self.m[key] = beta1 * self.m[key] + (1 - beta1) * grad

            # Update biased second moment estimate
            self.v[key] = beta2 * self.v[key] + (1 - beta2) * (grad ** 2)

            # Compute bias-corrected estimates
            m_hat = self.m[key] / (1 - beta1 ** step)
            v_hat = self.v[key] / (1 - beta2 ** step)

            # Update parameters with weight decay
            param_update = lr * m_hat / (np.sqrt(v_hat) + eps)
            param_update += lr * weight_decay * param  # Weight decay

            updated_params[key] = param - param_update

        return updated_params
```

### 6.2 Regularization Techniques

**Dropout**
- Random unit deactivation during training
- Prevents co-adaptation of features
- Applied to attention and feed-forward layers

**Label Smoothing**
```
smoothed_labels = (1 - α) * one_hot_labels + α * uniform_distribution
```

**Gradient Clipping**
```python
def clip_gradients(grads: Dict[str, np.ndarray], max_norm: float) -> Dict[str, np.ndarray]:
    """
    Clip gradients by norm

    Args:
        grads: Gradients dictionary
        max_norm: Maximum gradient norm

    Returns:
        clipped_grads: Clipped gradients
    """
    # Calculate global norm
    global_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads.values()))

    # Clip if necessary
    if global_norm > max_norm:
        scale = max_norm / global_norm
        clipped_grads = {k: v * scale for k, v in grads.items()}
    else:
        clipped_grads = grads

    return clipped_grads
```

## 7. Theoretical Analysis

### 7.1 Computational Complexity

**Transformer Complexity Analysis**

**Attention Mechanism**
- **Time Complexity**: O(n² × d_k) where n is sequence length
- **Space Complexity**: O(n²) for attention matrix
- **Memory Bandwidth**: O(n² × d_k) operations

**Feed-Forward Network**
- **Time Complexity**: O(n × d_model × d_ff)
- **Space Complexity**: O(n × d_ff)
- **Parameter Count**: 2 × d_model × d_ff + d_model + d_ff

**Overall Model**
- **Parameters**: O(d_model² × num_layers)
- **Memory**: O(n × d_model × num_layers)
- **Computation**: O(n² × d_model × num_layers)

**Efficiency Optimizations**
- **Sparse Attention**: Reduce O(n²) complexity
- **Linear Attention**: Approximate attention with linear complexity
- **Memory-efficient Attention**: Reduce memory usage for long sequences

### 7.2 Expressive Power

**Universal Approximation**
- Transformers can approximate any sequence-to-sequence function
- With sufficient width and depth
- Similar to neural network universal approximation theorem

**Representational Capacity**
- **Self-Attention**: Can represent arbitrary permutations
- **Multi-Head Attention**: Can learn different attention patterns
- **Position-wise FFN**: Can learn complex non-linear transformations

**Theoretical Limits**
- **Context Window**: Limited by quadratic attention complexity
- **Long-range Dependencies**: Despite global attention, practical limitations exist
- **Inductive Bias**: Less structured than RNNs

### 7.3 Generalization Properties

**Double Descent Phenomenon**
- Performance improves beyond interpolation threshold
- Related to overparameterization
- Observed in large language models

**Scaling Laws**
- **Performance**: Improves with model size, data, and compute
- **Power Law Relationships**: Systematic scaling behavior
- **Emergent Abilities**: New capabilities appear at scale

**Transfer Learning**
- **Pre-training**: Learn general language representations
- **Fine-tuning**: Adapt to specific tasks
- **Few-shot Learning**: Adapt with minimal examples

## 8. Advanced Topics

### 8.1 Attention Variants

**Sparse Attention Patterns**
- **Strided Attention**: Regular sparse patterns
- **Local Attention**: Focus on local context
- **Block Sparse Attention**: Hierarchical attention patterns

**Efficient Attention**
- **Linformer**: Linear complexity via low-rank approximation
- **Performer**: Kernel-based attention approximation
- **Reformer**: Locality-sensitive hashing for attention

**Hardware-aware Attention**
- **Flash Attention**: GPU-optimized attention
- **Memory-efficient Attention**: Reduce memory usage
- **Quantized Attention**: Low-precision computation

### 8.2 Architecture Variants

**Encoder-only Models**
- **BERT**: Bidirectional encoder representations
- **RoBERTa**: Optimized BERT training
- **ALBERT**: Parameter sharing for efficiency

**Decoder-only Models**
- **GPT Series**: Generative pre-trained transformers
- **LLaMA**: Open-source large language models
- **Claude**: Constitutional AI approach

**Encoder-Decoder Models**
- **T5**: Text-to-text transfer transformer
- **BART**: Denoising autoencoder
- **Pegasus**: Pre-training with gap sentences

### 8.3 Theoretical Extensions

**Graph Transformers**
- **Attention on Graphs**: Generalize to non-sequential data
- **Structural Information**: Incorporate graph topology
- **Applications**: Molecule modeling, social networks

**Multi-modal Transformers**
- **Vision-Transformers**: Apply attention to images
- **Audio Transformers**: Speech and music processing
- **Cross-modal Attention**: Different modalities

**Continuous Transformers**
- **Continuous Time**: Model continuous sequences
- **Neural ODEs**: Differential equation formulations
- **Infinite-depth**: Limit of infinitely many layers

## 9. Mathematical Insights

### 9.1 Optimization Landscape

**Loss Surface Properties**
- **Non-convexity**: Highly non-convex optimization
- **Saddle Points**: Many saddle points in high dimensions
- **Local Minima**: Numerous local minima of varying quality

**Gradient Dynamics**
- **Gradient Flow**: Continuous-time optimization
- **Stochasticity**: Random sampling effects
- **Momentum**: Accelerated optimization

### 9.2 Information Theory

**Information Bottleneck**
- **Compression**: Reduce input information
- **Prediction**: Preserve relevant information
- **Trade-off**: Optimal balance between compression and prediction

**Mutual Information**
- **Layer-wise Information**: Information flow through network
- **Information Plane**: Visualization of learning dynamics
- **Critical Learning Period**: Early training importance

### 9.3 Statistical Learning Theory

**Generalization Bounds**
- **VC Dimension**: Theoretical generalization capacity
- **Rademacher Complexity**: Measure of function class complexity
- **PAC Learning**: Probably approximately correct learning

**Sample Complexity**
- **Training Data**: Requirements for good generalization
- **Overfitting**: Memorization vs. generalization
- **Data Efficiency**: Learning from limited data

## Conclusion

The Transformer architecture represents a fundamental advancement in neural network design for sequential data. Its mathematical foundations in attention mechanisms, combined with efficient parallelization and scalable architecture, have enabled the development of increasingly powerful language models. Understanding these theoretical foundations is crucial for both practitioners and researchers working in NLP and related fields.

Key takeaways include:
1. **Attention as the Core**: Self-attention provides flexible context modeling
2. **Parallelization Advantage**: Enables efficient training on modern hardware
3. **Scalability**: Architecture scales effectively to billions of parameters
4. **Expressive Power**: Capable of learning complex sequence relationships
5. **Theoretical Richness**: Deep connections to information theory and optimization

The continued evolution of Transformer architectures will likely involve innovations in efficiency, generalization, and application to new domains while building on these fundamental mathematical principles.