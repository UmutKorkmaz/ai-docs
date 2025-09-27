# Transformer-Based Generative Models: Practical Implementations

## 1. Introduction to Transformer Generative Models

### 1.1 Architecture Overview

**Decoder-Only Transformers**
- **GPT Architecture**: Autoregressive language modeling
- **Causal Masking**: Ensure tokens only attend to previous tokens
- **Scalability**: Scales to billions of parameters
- **Versatility**: Foundation for text generation, code, and multimodal tasks

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple, Dict, List
import tiktoken
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings('ignore')

class Config:
    """Configuration for transformer models"""
    def __init__(self,
                 vocab_size: int = 50257,
                 n_layer: int = 12,
                 n_head: int = 12,
                 n_embd: int = 768,
                 block_size: int = 1024,
                 dropout: float = 0.1,
                 bias: bool = True):
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.block_size = block_size
        self.dropout = dropout
        self.bias = bias

    def __repr__(self):
        return f"Config(vocab_size={self.vocab_size}, n_layer={self.n_layer}, n_head={self.n_head}, n_embd={self.n_embd})"

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention mechanism
    """
    def __init__(self, config: Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # Key, query, value projections for all heads
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # Flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def forward(self, x: torch.Tensor, layer_past: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.size()  # Batch size, sequence length, embedding dimensionality

        # Calculate query, key, values for all heads in batch
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # Reshape into multiple heads
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # Causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # Efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True
            )
        else:
            # Manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # Re-assemble all head outputs side by side

        # Output projection
        y = self.resid_dropout(self.c_proj(y))

        # Cache for generation
        present = torch.stack((k, v), dim=0) if layer_past is None else torch.cat((layer_past, torch.stack((k, v), dim=0)), dim=1)

        return y, present

class MLP(nn.Module):
    """
    Multi-Layer Perceptron for transformer blocks
    """
    def __init__(self, config: Config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    """
    Transformer block: communication followed by computation
    """
    def __init__(self, config: Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, layer_past: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self-attention with residual connection
        attn_output, present = self.attn(self.ln_1(x), layer_past=layer_past)
        x = x + attn_output

        # MLP with residual connection
        x = x + self.mlp(self.ln_2(x))

        return x, present

class GPT(nn.Module):
    """
    GPT Language Model
    """
    def __init__(self, config: Config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)
        # Apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # Report number of parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Number of parameters: {n_params:,}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None,
                layer_past: Optional[List[torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # Forward the GPT model itself
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # shape (1, t)

        # Token and position embeddings
        tok_emb = self.transformer.wte(idx)  # Token embeddings
        pos_emb = self.transformer.wpe(pos)  # Position embeddings
        x = self.transformer.drop(tok_emb + pos_emb)

        # Transformer blocks
        presents = [] if layer_past is None else layer_past
        for i, block in enumerate(self.transformer.h):
            past = presents[i] if layer_past is not None else None
            x, present = block(x, layer_past=past)
            presents.append(present)

        # Final layer norm and classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # If we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return {
            'logits': logits,
            'loss': loss,
            'presents': presents
        }

    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: Optional[int] = None, top_p: Optional[float] = None,
                 do_sample: bool = True) -> torch.Tensor:
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # If the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            # Forward the model to get the logits for the index in the sequence
            outputs = self.forward(idx_cond)
            logits = outputs['logits']

            # Pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature

            # Optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Apply nucleus sampling (top-p)
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                # Scatter sorted tensors to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')

            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)

            # Either sample from the distribution or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)

            # Append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
```

### 1.2 Training Infrastructure

**Training Pipeline**
```python
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import time
import os
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    """Training configuration"""
    # Model parameters
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    vocab_size: int = 50257
    block_size: int = 1024

    # Training parameters
    batch_size: int = 4
    gradient_accumulation_steps: int = 32
    learning_rate: float = 6e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    max_grad_norm: float = 1.0

    # Scheduler
    lr_decay_iters: int = 600000
    lr_decay_frac: float = 0.1
    warmup_iters: int = 2000
    min_lr: float = 6e-5

    # Evaluation
    eval_interval: int = 2000
    eval_iters: int = 200
    log_interval: int = 1

    # Checkpointing
    out_dir: str = "out"
    eval_only: bool = False
    always_save_checkpoint: bool = True
    init_from: str = "scratch"  # "scratch", "resume", or "gpt2"

    # Data
    dataset: str = "openwebtext"
    gradient_checkpointing: bool = False

    # Distributed training
    device: str = "cuda"
    compile: bool = True
    distributed: bool = False

class TextDataset(Dataset):
    """Simple text dataset for language modeling"""
    def __init__(self, data_path: str, block_size: int, tokenizer):
        self.tokenizer = tokenizer
        self.block_size = block_size

        # Load and tokenize data
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()

        tokens = tokenizer.encode(text)
        self.data = torch.tensor(tokens, dtype=torch.long)

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

class Trainer:
    """GPT training infrastructure"""
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.setup_training()

    def setup_training(self):
        """Setup model, data, and training environment"""
        # Create config
        model_config = Config(
            vocab_size=self.config.vocab_size,
            n_layer=self.config.n_layer,
            n_head=self.config.n_head,
            n_embd=self.config.n_embd,
            block_size=self.config.block_size
        )

        # Initialize model
        self.model = GPT(model_config)
        self.model.to(self.device)

        # Initialize tokenizer
        self.tokenizer = tiktoken.get_encoding("gpt2")

        # Setup data
        self.setup_data()

        # Setup optimizer
        self.setup_optimizer()

        # Setup distributed training if needed
        if self.config.distributed:
            self.setup_distributed()

    def setup_data(self):
        """Setup training and validation data"""
        # This is a simplified version - in practice you'd use proper datasets
        # For demonstration, we'll create dummy data
        train_data = torch.randint(0, self.config.vocab_size, (10000, self.config.block_size + 1))
        val_data = torch.randint(0, self.config.vocab_size, (1000, self.config.block_size + 1))

        self.train_dataset = torch.utils.data.TensorDataset(train_data[:, :-1], train_data[:, 1:])
        self.val_dataset = torch.utils.data.TensorDataset(val_data[:, :-1], val_data[:, 1:])

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4
        )

    def setup_optimizer(self):
        """Setup optimizer with weight decay"""
        # Separate parameters for weight decay
        decay_params = []
        nodecay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # Don't apply weight decay to bias and LayerNorm parameters
            if len(param.shape) == 1 or name.endswith(".bias") or name.endswith(".ln_f.weight") or name.endswith(".ln_1.weight") or name.endswith(".ln_2.weight"):
                nodecay_params.append(param)
            else:
                decay_params.append(param)

        # Create optimizer groups
        optim_groups = [
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2)
        )

    def setup_distributed(self):
        """Setup distributed training"""
        if self.config.distributed:
            self.model = DDP(self.model, device_ids=[self.device])

    def get_learning_rate(self, iter: int) -> float:
        """Learning rate schedule with warmup and cosine decay"""
        # 1) Linear warmup for warmup_iters steps
        if iter < self.config.warmup_iters:
            return self.config.learning_rate * iter / self.config.warmup_iters

        # 2) If iter > lr_decay_iters, return min learning rate
        if iter > self.config.lr_decay_iters:
            return self.config.min_lr

        # 3) In between, use cosine decay down to min learning rate
        decay_ratio = (iter - self.config.warmup_iters) / (self.config.lr_decay_iters - self.config.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return self.config.min_lr + coeff * (self.config.learning_rate - self.config.min_lr)

    def train_epoch(self, epoch: int, iter_num: int = 0) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_tokens = 0

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Forward pass
            outputs = self.model(inputs, targets=targets)
            loss = outputs['loss']

            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps

            # Backward pass
            loss.backward()

            # Gradient clipping
            if batch_idx % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Update learning rate
            lr = self.get_learning_rate(iter_num)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            # Statistics
            total_loss += loss.item() * inputs.size(0)
            total_tokens += inputs.size(0) * inputs.size(1)
            iter_num += 1

            # Logging
            if batch_idx % self.config.log_interval == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, LR: {lr:.6f}")

            # Evaluation
            if iter_num % self.config.eval_interval == 0:
                eval_metrics = self.evaluate()
                print(f"Evaluation metrics: {eval_metrics}")

        avg_loss = total_loss / total_tokens
        return {'loss': avg_loss, 'tokens_processed': total_tokens}

    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on validation set"""
        self.model.eval()
        total_loss = 0
        total_tokens = 0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs, targets=targets)
                loss = outputs['loss']

                total_loss += loss.item() * inputs.size(0)
                total_tokens += inputs.size(0) * inputs.size(1)

        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)

        return {
            'val_loss': avg_loss,
            'perplexity': perplexity
        }

    def train(self, num_epochs: int = 10):
        """Main training loop"""
        iter_num = 0
        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            print(f"Starting epoch {epoch}")

            # Train epoch
            train_metrics = self.train_epoch(epoch, iter_num)

            # Validation
            eval_metrics = self.evaluate()
            print(f"Epoch {epoch} validation: {eval_metrics}")

            # Save checkpoint
            if eval_metrics['val_loss'] < best_val_loss or self.config.always_save_checkpoint:
                best_val_loss = eval_metrics['val_loss']
                self.save_checkpoint(epoch, eval_metrics)

        print("Training completed!")

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }

        os.makedirs(self.config.out_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.config.out_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Checkpoint loaded from {checkpoint_path}")
        return checkpoint['epoch'], checkpoint['metrics']
```

## 2. Advanced Generation Techniques

### 2.1 Sampling Strategies

**Advanced Sampling Methods**
```python
class AdvancedSampler:
    """Advanced sampling strategies for text generation"""

    def __init__(self, model: GPT, tokenizer, device: str = 'cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()

    def nucleus_sampling(self, prompt: str, max_length: int = 100,
                        top_p: float = 0.9, temperature: float = 1.0) -> str:
        """
        Nucleus (top-p) sampling

        Args:
            prompt: Input prompt
            max_length: Maximum length of generated text
            top_p: Cumulative probability threshold
            temperature: Temperature parameter

        Returns:
            Generated text
        """
        # Encode prompt
        input_ids = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(self.device)

        with torch.no_grad():
            for _ in range(max_length):
                # Get model predictions
                outputs = self.model.forward(input_ids)
                logits = outputs['logits'][:, -1, :]

                # Apply temperature
                logits = logits / temperature

                # Convert to probabilities
                probs = F.softmax(logits, dim=-1)

                # Sort probabilities in descending order
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)

                # Find nucleus
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                nucleus_mask = cumulative_probs <= top_p

                # Ensure at least one token is selected
        nucleus_mask[..., 0] = True

        # Filter probabilities
        filtered_probs = sorted_probs * nucleus_mask
        filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)

        # Sample from filtered distribution
        next_token = torch.multinomial(filtered_probs, num_samples=1)
        next_token = sorted_indices.gather(1, next_token)

        # Append to input
        input_ids = torch.cat([input_ids, next_token], dim=1)

        # Check for end of sequence
        if next_token.item() == self.tokenizer.eot_token:
            break

        # Decode and return
        generated_text = self.tokenizer.decode(input_ids[0].tolist())
        return generated_text

    def beam_search(self, prompt: str, max_length: int = 100,
                   num_beams: int = 5, length_penalty: float = 1.0) -> str:
        """
        Beam search decoding

        Args:
            prompt: Input prompt
            max_length: Maximum length of generated text
            num_beams: Number of beams
            length_penalty: Length penalty factor

        Returns:
            Generated text
        """
        # Encode prompt
        input_ids = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(self.device)

        # Initialize beams
        beams = [(input_ids, 0.0)]  # (sequence, score)

        with torch.no_grad():
            for step in range(max_length):
                new_beams = []

                for seq, score in beams:
                    # Get model predictions
                    outputs = self.model.forward(seq)
                    logits = outputs['logits'][:, -1, :]

                    # Get top-k candidates
                    probs = F.softmax(logits, dim=-1)
                    top_k_probs, top_k_indices = torch.topk(probs, num_beams, dim=-1)

                    # Expand beams
                    for i in range(num_beams):
                        next_token = top_k_indices[:, i:i+1]
                        next_prob = top_k_probs[:, i:i+1]

                        new_seq = torch.cat([seq, next_token], dim=1)
                        new_score = score + torch.log(next_prob)

                        # Apply length penalty
                        new_score = new_score / (len(new_seq[0]) ** length_penalty)

                        new_beams.append((new_seq, new_score))

                # Keep top beams
                new_beams.sort(key=lambda x: x[1], reverse=True)
                beams = new_beams[:num_beams]

        # Return best beam
        best_seq, _ = beams[0]
        generated_text = self.tokenizer.decode(best_seq[0].tolist())
        return generated_text

    def contrastive_search(self, prompt: str, max_length: int = 100,
                          top_k: int = 4, alpha: float = 0.6) -> str:
        """
        Contrastive search for more diverse generation

        Args:
            prompt: Input prompt
            max_length: Maximum length
            top_k: Number of candidates to consider
            alpha: Balance between model confidence and degeneration penalty

        Returns:
            Generated text
        """
        # Encode prompt
        input_ids = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(self.device)

        # Store previous tokens for degeneration penalty
        prev_tokens = []

        with torch.no_grad():
            for _ in range(max_length):
                # Get model predictions
                outputs = self.model.forward(input_ids)
                logits = outputs['logits'][:, -1, :]

                # Get top-k candidates
                probs = F.softmax(logits, dim=-1)
                top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)

                # Calculate degeneration penalty
                scores = []
                for i in range(top_k):
                    token_id = top_k_indices[0, i].item()

                    # Model confidence
                    model_score = top_k_probs[0, i].item()

                    # Degeneration penalty
                    deg_penalty = 0.0
                    for prev_token in prev_tokens[-3:]:  # Check last 3 tokens
                        if token_id == prev_token:
                            deg_penalty += 1.0

                    # Combined score
                    combined_score = (1 - alpha) * model_score - alpha * deg_penalty
                    scores.append(combined_score)

                # Select best token
                best_idx = np.argmax(scores)
                next_token = top_k_indices[:, best_idx:best_idx+1]

                # Update input and previous tokens
                input_ids = torch.cat([input_ids, next_token], dim=1)
                prev_tokens.append(next_token.item())

                # Check for end of sequence
                if next_token.item() == self.tokenizer.eot_token:
                    break

        # Decode and return
        generated_text = self.tokenizer.decode(input_ids[0].tolist())
        return generated_text

    def generate_with_constraints(self, prompt: str, constraints: List[str],
                                max_length: int = 100) -> str:
        """
        Generate text with specific constraints

        Args:
            prompt: Input prompt
            constraints: List of required keywords/phrases
            max_length: Maximum length

        Returns:
            Generated text satisfying constraints
        """
        # Encode prompt
        input_ids = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(self.device)

        satisfied_constraints = set()
        generated_text = prompt

        with torch.no_grad():
            for _ in range(max_length):
                # Check if all constraints are satisfied
                if all(constraint.lower() in generated_text.lower() for constraint in constraints):
                    break

                # Get model predictions
                outputs = self.model.forward(input_ids)
                logits = outputs['logits'][:, -1, :]

                # Bias towards constraint tokens
                for constraint in constraints:
                    if constraint.lower() not in generated_text.lower():
                        constraint_tokens = self.tokenizer.encode(constraint)
                        for token in constraint_tokens:
                            logits[0, token] += 2.0  # Boost constraint tokens

                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Update input and generated text
                input_ids = torch.cat([input_ids, next_token], dim=1)
                generated_text = self.tokenizer.decode(input_ids[0].tolist())

        return generated_text

    def iterative_refinement(self, prompt: str, max_length: int = 100,
                           num_iterations: int = 3) -> str:
        """
        Iterative refinement for higher quality generation

        Args:
            prompt: Input prompt
            max_length: Maximum length
            num_iterations: Number of refinement iterations

        Returns:
            Refined generated text
        """
        best_text = prompt
        best_score = float('-inf')

        for iteration in range(num_iterations):
            # Generate text
            generated_text = self.nucleus_sampling(
                best_text,
                max_length=max_length,
                temperature=0.7 + 0.1 * iteration  # Vary temperature
            )

            # Score the generated text
            score = self._score_generation(generated_text, prompt)

            if score > best_score:
                best_text = generated_text
                best_score = score

        return best_text

    def _score_generation(self, generated_text: str, prompt: str) -> float:
        """
        Score generation quality (simplified)

        Args:
            generated_text: Generated text
            prompt: Original prompt

        Returns:
            Quality score
        """
        # Encode text
        input_ids = torch.tensor(self.tokenizer.encode(generated_text), dtype=torch.long).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model.forward(input_ids)
            logits = outputs['logits']

            # Calculate perplexity
            target_ids = input_ids[:, 1:]
            pred_logits = logits[:, :-1, :]

            loss = F.cross_entropy(pred_logits.view(-1, pred_logits.size(-1)),
                                  target_ids.view(-1), ignore_index=-1)
            perplexity = torch.exp(loss)

        # Lower perplexity is better
        return -perplexity.item()
```

### 2.2 Controlled Generation

**Controllable Text Generation**
```python
class ControlledGenerator:
    """Controlled text generation with various techniques"""

    def __init__(self, model: GPT, tokenizer, device: str = 'cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()

    def generate_with_attributes(self, prompt: str, attributes: Dict[str, float],
                              max_length: int = 100) -> str:
        """
        Generate text with specific attributes (sentiment, style, etc.)

        Args:
            prompt: Input prompt
            attributes: Dictionary of attribute values
            max_length: Maximum length

        Returns:
            Generated text with desired attributes
        """
        # Encode prompt
        input_ids = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(self.device)

        # Attribute control vectors (simplified)
        attribute_controls = self._get_attribute_controls(attributes)

        with torch.no_grad():
            for _ in range(max_length):
                # Get model predictions
                outputs = self.model.forward(input_ids)
                logits = outputs['logits'][:, -1, :]

                # Apply attribute controls
                for attr_name, attr_value in attributes.items():
                    if attr_name in attribute_controls:
                        control_vector = attribute_controls[attr_name]
                        logits += attr_value * control_vector

                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Update input
                input_ids = torch.cat([input_ids, next_token], dim=1)

                # Check for end of sequence
                if next_token.item() == self.tokenizer.eot_token:
                    break

        # Decode and return
        generated_text = self.tokenizer.decode(input_ids[0].tolist())
        return generated_text

    def _get_attribute_controls(self, attributes: Dict[str, float]) -> Dict[str, torch.Tensor]:
        """
        Get attribute control vectors (simplified)

        Args:
            attributes: Attribute dictionary

        Returns:
            Control vectors for each attribute
        """
        # This is a simplified version
        # In practice, you'd train these or use more sophisticated methods

        control_vectors = {}

        # Sentiment control
        if 'sentiment' in attributes:
            sentiment = attributes['sentiment']
            if sentiment > 0:  # Positive
                # Boost positive words
                positive_tokens = self.tokenizer.encode(" good great amazing wonderful")
                control_vector = torch.zeros(self.model.config.vocab_size)
                for token in positive_tokens:
                    control_vector[token] = 1.0
                control_vectors['sentiment'] = control_vector.to(self.device)
            else:  # Negative
                # Boost negative words
                negative_tokens = self.tokenizer.encode(" bad terrible awful horrible")
                control_vector = torch.zeros(self.model.config.vocab_size)
                for token in negative_tokens:
                    control_vector[token] = 1.0
                control_vectors['sentiment'] = control_vector.to(self.device)

        return control_vectors

    def generate_with_topic_control(self, prompt: str, topic: str,
                                   max_length: int = 100) -> str:
        """
        Generate text about a specific topic

        Args:
            prompt: Input prompt
            topic: Desired topic
            max_length: Maximum length

        Returns:
            Text focused on the specified topic
        """
        # Encode prompt
        input_ids = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(self.device)

        # Get topic tokens
        topic_tokens = set(self.tokenizer.encode(topic))

        with torch.no_grad():
            for _ in range(max_length):
                # Get model predictions
                outputs = self.model.forward(input_ids)
                logits = outputs['logits'][:, -1, :]

                # Boost topic-related tokens
                for token in topic_tokens:
                    logits[0, token] += 1.0

                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Update input
                input_ids = torch.cat([input_ids, next_token], dim=1)

                # Check for end of sequence
                if next_token.item() == self.tokenizer.eot_token:
                    break

        # Decode and return
        generated_text = self.tokenizer.decode(input_ids[0].tolist())
        return generated_text

    def generate_with_length_control(self, prompt: str, target_length: int,
                                    max_attempts: int = 10) -> str:
        """
        Generate text of specific length

        Args:
            prompt: Input prompt
            target_length: Target length in tokens
            max_attempts: Maximum generation attempts

        Returns:
            Text close to target length
        """
        best_text = prompt
        best_diff = float('inf')

        for attempt in range(max_attempts):
            # Generate text with varying parameters
            if attempt < max_attempts // 2:
                # Use beam search for shorter text
                generated_text = self.sampler.beam_search(
                    prompt, max_length=target_length, num_beams=3
                )
            else:
                # Use sampling for more natural text
                generated_text = self.sampler.nucleus_sampling(
                    prompt, max_length=target_length, top_p=0.9
                )

            # Calculate length difference
            generated_tokens = self.tokenizer.encode(generated_text)
            length_diff = abs(len(generated_tokens) - target_length)

            if length_diff < best_diff:
                best_text = generated_text
                best_diff = length_diff

                # Early stopping if close enough
                if length_diff <= 5:  # Within 5 tokens
                    break

        return best_text

    def generate_with_repetition_penalty(self, prompt: str, penalty_factor: float = 1.2,
                                       max_length: int = 100) -> str:
        """
        Generate text with repetition penalty

        Args:
            prompt: Input prompt
            penalty_factor: Repetition penalty factor
            max_length: Maximum length

        Returns:
            Text with reduced repetition
        """
        # Encode prompt
        input_ids = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(self.device)

        with torch.no_grad():
            for _ in range(max_length):
                # Get model predictions
                outputs = self.model.forward(input_ids)
                logits = outputs['logits'][:, -1, :]

                # Apply repetition penalty
                for token_id in input_ids[0].unique():
                    logits[0, token_id] /= penalty_factor

                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Update input
                input_ids = torch.cat([input_ids, next_token], dim=1)

                # Check for end of sequence
                if next_token.item() == self.tokenizer.eot_token:
                    break

        # Decode and return
        generated_text = self.tokenizer.decode(input_ids[0].tolist())
        return generated_text

    def interactive_generation(self, initial_prompt: str, max_length: int = 1000):
        """
        Interactive text generation with user feedback

        Args:
            initial_prompt: Starting prompt
            max_length: Maximum total length
        """
        current_text = initial_prompt
        print(f"Initial prompt: {initial_prompt}")
        print("Type 'continue' to generate more, 'undo' to remove last part, or 'quit' to exit")

        while True:
            user_input = input("> ").lower().strip()

            if user_input == 'quit':
                break
            elif user_input == 'continue':
                # Generate next part
                continuation = self.sampler.nucleus_sampling(
                    current_text, max_length=50, top_p=0.9
                )
                current_text = continuation
                print(f"\nGenerated: {current_text}")
            elif user_input == 'undo':
                # Remove last part (simplified)
                words = current_text.split()
                if len(words) > 10:
                    current_text = ' '.join(words[:-10])
                    print(f"\nCurrent text: {current_text}")
                else:
                    print("Cannot undo further")
            else:
                # Treat as new prompt
                current_text = user_input
                print(f"\nNew prompt: {current_text}")

        print(f"\nFinal text: {current_text}")

    def batch_generation(self, prompts: List[str], generation_params: Dict,
                        batch_size: int = 8) -> List[str]:
        """
        Generate text for multiple prompts in batch

        Args:
            prompts: List of input prompts
            generation_params: Generation parameters
            batch_size: Batch size for processing

        Returns:
            List of generated texts
        """
        results = []

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]

            # Tokenize batch
            batch_input_ids = []
            for prompt in batch_prompts:
                input_ids = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.long)
                batch_input_ids.append(input_ids)

            # Pad sequences
            max_len = max(len(ids) for ids in batch_input_ids)
            padded_input_ids = torch.zeros((len(batch_input_ids), max_len), dtype=torch.long)
            for j, ids in enumerate(batch_input_ids):
                padded_input_ids[j, :len(ids)] = ids

            padded_input_ids = padded_input_ids.to(self.device)

            # Generate in batch
            with torch.no_grad():
                for _ in range(generation_params.get('max_new_tokens', 100)):
                    outputs = self.model.forward(padded_input_ids)
                    logits = outputs['logits'][:, -1, :]

                    # Apply sampling strategy
                    if generation_params.get('do_sample', True):
                        probs = F.softmax(logits / generation_params.get('temperature', 1.0), dim=-1)
                        next_tokens = torch.multinomial(probs, num_samples=1)
                    else:
                        next_tokens = torch.argmax(logits, dim=-1, keepdim=True)

                    # Append to input
                    padded_input_ids = torch.cat([padded_input_ids, next_tokens], dim=1)

                    # Check if all sequences ended
                    if all(token == self.tokenizer.eot_token for token in next_tokens.squeeze()):
                        break

            # Decode results
            for j in range(len(batch_prompts)):
                generated_text = self.tokenizer.decode(padded_input_ids[j].tolist())
                results.append(generated_text)

        return results

    def generate_with_memory(self, prompt: str, memory_context: List[str],
                           max_length: int = 100) -> str:
        """
        Generate text with memory/context awareness

        Args:
            prompt: Current prompt
            memory_context: List of previous context
            max_length: Maximum length

        Returns:
            Context-aware generated text
        """
        # Combine memory with current prompt
        if memory_context:
            context_prompt = "\n".join(memory_context[-3:]) + "\n" + prompt  # Use last 3 memories
        else:
            context_prompt = prompt

        # Generate with enhanced context
        generated_text = self.sampler.nucleus_sampling(
            context_prompt, max_length=max_length, top_p=0.9
        )

        return generated_text
```

## 3. Fine-Tuning and Adaptation

### 3.1 Parameter-Efficient Fine-Tuning

**LoRA and QLoRA Implementation**
```python
import torch.nn.utils.parametrize as parametrize
from typing import Optional, Tuple

class LoRALayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) layer
    """
    def __init__(self, original_layer: nn.Linear, rank: int = 8, alpha: float = 16):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha

        # Freeze original parameters
        for param in self.original_layer.parameters():
            param.requires_grad = False

        # Low-rank matrices
        in_features = original_layer.in_features
        out_features = original_layer.out_features

        self.lora_A = nn.Parameter(torch.randn(rank, in_features))
        self.lora_B = nn.Parameter(torch.randn(out_features, rank))
        self.scaling = alpha / rank

        # Initialize LoRA matrices
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original layer output
        original_output = self.original_layer(x)

        # LoRA adaptation
        lora_output = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling

        return original_output + lora_output

class QLoRALayer(nn.Module):
    """
    Quantized LoRA layer
    """
    def __init__(self, original_layer: nn.Linear, rank: int = 8, alpha: float = 16,
                 quantization_bits: int = 4):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.quant_bits = quantization_bits

        # Freeze and quantize original layer
        self._quantize_original_layer()

        # Low-rank matrices
        in_features = original_layer.in_features
        out_features = original_layer.out_features

        self.lora_A = nn.Parameter(torch.randn(rank, in_features))
        self.lora_B = nn.Parameter(torch.randn(out_features, rank))
        self.scaling = alpha / rank

        # Initialize LoRA matrices
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def _quantize_original_layer(self):
        """Quantize the original layer weights"""
        weight = self.original_layer.weight.data
        quantized_weight = self._quantize(weight, self.quant_bits)
        self.original_layer.weight.data = quantized_weight

    def _quantize(self, tensor: torch.Tensor, bits: int) -> torch.Tensor:
        """Simple uniform quantization"""
        qmin = -2 ** (bits - 1)
        qmax = 2 ** (bits - 1) - 1

        scale = (tensor.max() - tensor.min()) / (qmax - qmin)
        zero_point = qmin - tensor.min() / scale

        quantized = torch.clamp((tensor / scale + zero_point).round(), qmin, qmax)
        return (quantized - zero_point) * scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original layer output (quantized)
        original_output = self.original_layer(x)

        # LoRA adaptation
        lora_output = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling

        return original_output + lora_output

class AdapterLayer(nn.Module):
    """
    Adapter layer for parameter-efficient fine-tuning
    """
    def __init__(self, input_dim: int, bottleneck_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim

        # Adapter layers
        self.adapter_down = nn.Linear(input_dim, bottleneck_dim)
        self.adapter_up = nn.Linear(bottleneck_dim, input_dim)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(input_dim)

        # Initialize adapter weights
        nn.init.kaiming_uniform_(self.adapter_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.adapter_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Adapter forward pass
        residual = x
        x = self.layer_norm(x)
        x = self.adapter_down(x)
        x = self.activation(x)
        x = self.adapter_up(x)

        return residual + x

class PrefixTuning(nn.Module):
    """
    Prefix-tuning for controllable generation
    """
    def __init__(self, model: GPT, prefix_length: int = 10):
        super().__init__()
        self.model = model
        self.prefix_length = prefix_length
        self.n_layer = model.config.n_layer
        self.n_head = model.config.n_head
        self.n_embd = model.config.n_embd

        # Prefix parameters
        self.prefix_tokens = nn.Parameter(torch.randn(prefix_length, model.config.n_embd))

        # Prefix projection matrices
        self.w_key = nn.Linear(model.config.n_embd, model.config.n_embd)
        self.w_value = nn.Linear(model.config.n_embd, model.config.n_embd)

        # Freeze model parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        batch_size = input_ids.size(0)

        # Generate prefix keys and values
        prefix_keys = self.w_key(self.prefix_tokens)  # (prefix_length, n_embd)
        prefix_values = self.w_value(self.prefix_tokens)  # (prefix_length, n_embd)

        # Repeat for batch
        prefix_keys = prefix_keys.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, prefix_length, n_embd)
        prefix_values = prefix_values.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, prefix_length, n_embd)

        # Combine with input
        input_embeds = self.model.transformer.wte(input_ids)
        prefix_embeds = self.model.transformer.wpe(torch.arange(self.prefix_length, device=input_ids.device))
        prefix_embeds = prefix_embeds.unsqueeze(0).expand(batch_size, -1, -1)

        # Combine embeddings
        combined_embeds = torch.cat([prefix_embeds, input_embeds], dim=1)

        # Add position embeddings
        pos = torch.arange(0, combined_embeds.size(1), dtype=torch.long, device=input_ids.device)
        pos_embeds = self.model.transformer.wpe(pos)
        x = combined_embeds + pos_embeds

        # Transformer forward pass with modified attention
        presents = []
        for i, block in enumerate(self.model.transformer.h):
            # Modify attention to include prefix
            x, present = block._attn_with_prefix(x, prefix_keys, prefix_values, i)
            presents.append(present)

        # Final layers
        x = self.model.transformer.ln_f(x)
        logits = self.model.lm_head(x)

        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            # Adjust targets for prefix
            prefix_targets = torch.full((batch_size, self.prefix_length), -100, device=input_ids.device)
            adjusted_targets = torch.cat([prefix_targets, targets], dim=1)

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                adjusted_targets.view(-1),
                ignore_index=-100
            )

        return {
            'logits': logits,
            'loss': loss,
            'presents': presents
        }

class P Tuning(nn.Module):
    """
    P-tuning using continuous prompt embeddings
    """
    def __init__(self, model: GPT, prompt_length: int = 10):
        super().__init__()
        self.model = model
        self.prompt_length = prompt_length

        # Learnable prompt embeddings
        self.prompt_embeddings = nn.Parameter(torch.randn(prompt_length, model.config.n_embd))

        # Freeze model parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        batch_size = input_ids.size(0)

        # Get input embeddings
        input_embeds = self.model.transformer.wte(input_ids)

        # Get prompt embeddings
        prompt_embeds = self.prompt_embeddings.unsqueeze(0).expand(batch_size, -1, -1)

        # Combine embeddings
        combined_embeds = torch.cat([prompt_embeds, input_embeds], dim=1)

        # Add position embeddings
        total_length = combined_embeds.size(1)
        pos = torch.arange(0, total_length, dtype=torch.long, device=input_ids.device)
        pos_embeds = self.model.transformer.wpe(pos)
        x = combined_embeds + pos_embeds

        # Transformer forward pass
        presents = []
        for block in self.model.transformer.h:
            x, present = block(x)
            presents.append(present)

        # Final layers
        x = self.model.transformer.ln_f(x)
        logits = self.model.lm_head(x)

        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            # Adjust targets for prompt
            prompt_targets = torch.full((batch_size, self.prompt_length), -100, device=input_ids.device)
            adjusted_targets = torch.cat([prompt_targets, targets], dim=1)

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                adjusted_targets.view(-1),
                ignore_index=-100
            )

        return {
            'logits': logits,
            'loss': loss,
            'presents': presents
        }

class EfficientFineTuner:
    """Efficient fine-tuning framework"""

    def __init__(self, model: GPT):
        self.model = model
        self.original_state_dict = None

    def apply_lora(self, rank: int = 8, alpha: float = 16,
                  target_modules: List[str] = None) -> GPT:
        """
        Apply LoRA to specified modules

        Args:
            rank: LoRA rank
            alpha: LoRA alpha parameter
            target_modules: List of module names to apply LoRA

        Returns:
            Model with LoRA applied
        """
        if target_modules is None:
            target_modules = ['attn.c_attn', 'attn.c_proj', 'mlp.c_fc', 'mlp.c_proj']

        # Store original state
        self.original_state_dict = self.model.state_dict()

        # Apply LoRA to target modules
        for name, module in self.model.named_modules():
            if any(target in name for target in target_modules):
                if isinstance(module, nn.Linear):
                    # Replace with LoRA layer
                    parent_name, child_name = name.rsplit('.', 1)
                    parent = self.model.get_submodule(parent_name)
                    setattr(parent, child_name, LoRALayer(module, rank, alpha))

        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"LoRA applied: {trainable_params:,} trainable parameters ({100*trainable_params/total_params:.2f}%)")

        return self.model

    def apply_qlora(self, rank: int = 8, alpha: float = 16, quant_bits: int = 4,
                   target_modules: List[str] = None) -> GPT:
        """
        Apply QLoRA to specified modules

        Args:
            rank: LoRA rank
            alpha: LoRA alpha parameter
            quant_bits: Quantization bits
            target_modules: List of module names to apply QLoRA

        Returns:
            Model with QLoRA applied
        """
        if target_modules is None:
            target_modules = ['attn.c_attn', 'attn.c_proj', 'mlp.c_fc', 'mlp.c_proj']

        # Store original state
        self.original_state_dict = self.model.state_dict()

        # Apply QLoRA to target modules
        for name, module in self.model.named_modules():
            if any(target in name for target in target_modules):
                if isinstance(module, nn.Linear):
                    # Replace with QLoRA layer
                    parent_name, child_name = name.rsplit('.', 1)
                    parent = self.model.get_submodule(parent_name)
                    setattr(parent, child_name, QLoRALayer(module, rank, alpha, quant_bits))

        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"QLoRA applied: {trainable_params:,} trainable parameters ({100*trainable_params/total_params:.2f}%)")

        return self.model

    def apply_adapters(self, bottleneck_dim: int = 64,
                      target_modules: List[str] = None) -> GPT:
        """
        Apply adapters to specified modules

        Args:
            bottleneck_dim: Bottleneck dimension for adapters
            target_modules: List of module names to apply adapters

        Returns:
            Model with adapters applied
        """
        if target_modules is None:
            target_modules = ['ln_1', 'ln_2']  # Apply after layer norms

        # Store original state
        self.original_state_dict = self.model.state_dict()

        # Apply adapters
        for name, module in self.model.named_modules():
            if any(target in name for target in target_modules):
                if isinstance(module, nn.LayerNorm):
                    # Add adapter after layer norm
                    parent_name, child_name = name.rsplit('.', 1)
                    parent = self.model.get_submodule(parent_name)

                    # Create adapter module
                    adapter = AdapterLayer(module.normalized_shape[0], bottleneck_dim)

                    # Create sequential module
                    sequential = nn.Sequential(module, adapter)
                    setattr(parent, child_name, sequential)

        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Adapters applied: {trainable_params:,} trainable parameters ({100*trainable_params/total_params:.2f}%)")

        return self.model

    def restore_original_model(self) -> GPT:
        """
        Restore original model without adaptations

        Returns:
            Original model
        """
        if self.original_state_dict is not None:
            self.model.load_state_dict(self.original_state_dict)
            print("Original model restored")

        return self.model

    def train_with_adaptation(self, train_loader: DataLoader, val_loader: DataLoader,
                            num_epochs: int = 10, learning_rate: float = 1e-3) -> Dict[str, List[float]]:
        """
        Train model with adaptations

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            learning_rate: Learning rate

        Returns:
            Training history
        """
        # Setup optimizer (only for trainable parameters)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)

        # Training history
        history = {'train_loss': [], 'val_loss': [], 'val_perplexity': []}

        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                optimizer.zero_grad()

                outputs = self.model.forward(inputs, targets=targets)
                loss = outputs['loss']

                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Validation phase
            self.model.eval()
            val_loss = 0
            val_perplexity = 0

            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    outputs = self.model.forward(inputs, targets=targets)
                    loss = outputs['loss']
                    perplexity = torch.exp(loss)

                    val_loss += loss.item()
                    val_perplexity += perplexity.item()

            # Calculate averages
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            val_perplexity /= len(val_loader)

            # Store history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_perplexity'].append(val_perplexity)

            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Perplexity: {val_perplexity:.4f}")
            print("-" * 50)

        return history

    def merge_adaptations(self) -> GPT:
        """
        Merge LoRA/adapters back into original model

        Returns:
            Model with merged adaptations
        """
        for name, module in self.model.named_modules():
            if isinstance(module, (LoRALayer, QLoRALayer)):
                # Merge LoRA weights
                original_weight = module.original_layer.weight.data
                lora_weight = module.lora_B @ module.lora_A * module.scaling
                merged_weight = original_weight + lora_weight.T

                # Replace with standard linear layer
                parent_name, child_name = name.rsplit('.', 1)
                parent = self.model.get_submodule(parent_name)

                new_linear = nn.Linear(
                    original_weight.size(1),
                    original_weight.size(0),
                    bias=module.original_layer.bias is not None
                )
                new_linear.weight.data = merged_weight
                if module.original_layer.bias is not None:
                    new_linear.bias.data = module.original_layer.bias.data

                setattr(parent, child_name, new_linear)

            elif isinstance(module, nn.Sequential):
                # Check if this is an adapter + layer norm
                if len(module) == 2 and isinstance(module[0], nn.LayerNorm) and isinstance(module[1], AdapterLayer):
                    # Replace with just layer norm (adapter effects are not typically merged)
                    parent_name, child_name = name.rsplit('.', 1)
                    parent = self.model.get_submodule(parent_name)
                    setattr(parent, child_name, module[0])

        print("Adaptations merged into model")
        return self.model
```

### 3.2 Instruction Fine-Tuning

**Instruction Following and RLHF**
```python
class InstructionDataset(Dataset):
    """
    Dataset for instruction fine-tuning
    """
    def __init__(self, instructions: List[Dict], tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.instructions = instructions

    def __len__(self):
        return len(self.instructions)

    def __getitem__(self, idx):
        instruction = self.instructions[idx]

        # Format instruction
        prompt = instruction['instruction']
        if 'input' in instruction and instruction['input']:
            prompt += f"\nInput: {instruction['input']}"

        prompt += "\nOutput:"

        # Tokenize
        full_text = prompt + " " + instruction['output']
        tokens = self.tokenizer.encode(full_text)

        # Truncate if too long
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]

        # Create input and target sequences
        input_text = prompt
        input_tokens = self.tokenizer.encode(input_text)
        target_tokens = tokens[len(input_tokens):]

        # Pad sequences
        input_tokens = input_tokens + [-100] * (self.max_length - len(input_tokens))
        target_tokens = target_tokens + [-100] * (self.max_length - len(target_tokens))

        return {
            'input_ids': torch.tensor(input_tokens[:self.max_length], dtype=torch.long),
            'labels': torch.tensor(target_tokens[:self.max_length], dtype=torch.long),
            'attention_mask': torch.tensor([1] * min(len(input_tokens), self.max_length), dtype=torch.long)
        }

class RewardModel(nn.Module):
    """
    Reward model for RLHF
    """
    def __init__(self, base_model: GPT, reward_head_dim: int = 128):
        super().__init__()
        self.base_model = base_model
        self.reward_head = nn.Sequential(
            nn.Linear(base_model.config.n_embd, reward_head_dim),
            nn.ReLU(),
            nn.Linear(reward_head_dim, 1)
        )

        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Get base model embeddings
        outputs = self.base_model.forward(input_ids)
        last_hidden_state = outputs['logits']  # Get last hidden state

        # Use last token representation for reward
        reward_input = last_hidden_state[:, -1, :]
        reward = self.reward_head(reward_input)

        return reward.squeeze(-1)

class RLHFTrainer:
    """
    Reinforcement Learning from Human Feedback trainer
    """
    def __init__(self, policy_model: GPT, reward_model: RewardModel,
                 reference_model: GPT, tokenizer, kl_coef: float = 0.1):
        self.policy_model = policy_model
        self.reward_model = reward_model
        self.reference_model = reference_model
        self.tokenizer = tokenizer
        self.kl_coef = kl_coef

        # Freeze reference model
        for param in self.reference_model.parameters():
            param.requires_grad = False

        # Setup optimizer
        self.optimizer = torch.optim.Adam(
            [p for p in self.policy_model.parameters() if p.requires_grad],
            lr=1e-5
        )

    def generate_responses(self, prompts: List[str], generation_kwargs: Dict) -> List[str]:
        """
        Generate responses from policy model
        """
        responses = []

        for prompt in prompts:
            # Encode prompt
            input_ids = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0)

            # Generate response
            with torch.no_grad():
                generated_ids = self.policy_model.generate(input_ids, **generation_kwargs)

            # Decode
            response = self.tokenizer.decode(generated_ids[0].tolist())
            responses.append(response)

        return responses

    def compute_rewards(self, prompts: List[str], responses: List[str]) -> torch.Tensor:
        """
        Compute rewards using reward model
        """
        rewards = []

        for prompt, response in zip(prompts, responses):
            # Combine prompt and response
            full_text = prompt + " " + response
            input_ids = torch.tensor(self.tokenizer.encode(full_text), dtype=torch.long).unsqueeze(0)

            # Get reward
            with torch.no_grad():
                reward = self.reward_model(input_ids)
                rewards.append(reward.item())

        return torch.tensor(rewards)

    def compute_kl_penalty(self, prompts: List[str], responses: List[str]) -> torch.Tensor:
        """
        Compute KL divergence penalty between policy and reference models
        """
        kl_penalties = []

        for prompt, response in zip(prompts, responses):
            # Combine prompt and response
            full_text = prompt + " " + response
            input_ids = torch.tensor(self.tokenizer.encode(full_text), dtype=torch.long).unsqueeze(0)

            # Get logits from both models
            with torch.no_grad():
                policy_outputs = self.policy_model.forward(input_ids)
                ref_outputs = self.reference_model.forward(input_ids)

                policy_probs = F.softmax(policy_outputs['logits'], dim=-1)
                ref_probs = F.softmax(ref_outputs['logits'], dim=-1)

                # Compute KL divergence
                kl = F.kl_div(
                    torch.log(policy_probs + 1e-10),
                    ref_probs,
                    reduction='batchmean'
                )
                kl_penalties.append(kl.item())

        return torch.tensor(kl_penalties)

    def ppo_update(self, prompts: List[str], responses: List[str],
                   rewards: torch.Tensor, advantages: torch.Tensor,
                   epochs: int = 4, batch_size: int = 32) -> Dict[str, float]:
        """
        PPO update step
        """
        all_metrics = []

        for epoch in range(epochs):
            # Shuffle data
            indices = torch.randperm(len(prompts))
            shuffled_prompts = [prompts[i] for i in indices]
            shuffled_responses = [responses[i] for i in indices]
            shuffled_rewards = rewards[indices]
            shuffled_advantages = advantages[indices]

            epoch_metrics = {'loss': [], 'policy_loss': [], 'value_loss': [], 'entropy': []}

            # Process in batches
            for i in range(0, len(prompts), batch_size):
                batch_prompts = shuffled_prompts[i:i+batch_size]
                batch_responses = shuffled_responses[i:i+batch_size]
                batch_rewards = shuffled_rewards[i:i+batch_size]
                batch_advantages = shuffled_advantages[i:i+batch_size]

                # Forward pass
                batch_loss, batch_metrics = self._compute_ppo_loss(
                    batch_prompts, batch_responses, batch_rewards, batch_advantages
                )

                # Backward pass
                self.optimizer.zero_grad()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
                self.optimizer.step()

                # Store metrics
                for key, value in batch_metrics.items():
                    epoch_metrics[key].append(value)

            # Average metrics
            avg_metrics = {key: np.mean(values) for key, values in epoch_metrics.items()}
            all_metrics.append(avg_metrics)

        return all_metrics

    def _compute_ppo_loss(self, prompts: List[str], responses: List[str],
                          rewards: torch.Tensor, advantages: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute PPO loss for a batch
        """
        # Get old policy log probabilities
        old_log_probs = self._get_log_probs(prompts, responses, self.policy_model)

        # Compute KL penalty
        kl_penalties = self.compute_kl_penalty(prompts, responses)

        # Compute loss
        ratio = torch.exp(old_log_probs - old_log_probs.detach())  # Simplified
        policy_loss = -torch.min(
            ratio * advantages,
            torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * advantages
        ).mean()

        # KL penalty
        kl_loss = self.kl_coef * kl_penalties.mean()

        # Total loss
        total_loss = policy_loss + kl_loss

        # Compute entropy
        with torch.no_grad():
            probs = torch.exp(old_log_probs)
            entropy = -(probs * old_log_probs).sum(dim=-1).mean()

        metrics = {
            'loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'kl_loss': kl_loss.item(),
            'entropy': entropy.item()
        }

        return total_loss, metrics

    def _get_log_probs(self, prompts: List[str], responses: List[str], model: GPT) -> torch.Tensor:
        """
        Get log probabilities from model
        """
        log_probs = []

        for prompt, response in zip(prompts, responses):
            # Combine prompt and response
            full_text = prompt + " " + response
            input_ids = torch.tensor(self.tokenizer.encode(full_text), dtype=torch.long).unsqueeze(0)

            # Get model outputs
            with torch.no_grad():
                outputs = model.forward(input_ids)
                logits = outputs['logits']

                # Get log probabilities
                log_probs_seq = F.log_softmax(logits, dim=-1)

                # Get log probs for actual tokens
                response_ids = torch.tensor(self.tokenizer.encode(response), dtype=torch.long)
                response_log_probs = log_probs_seq[0, len(self.tokenizer.encode(prompt)):len(response_ids)]

                log_probs.append(response_log_probs.sum().item())

        return torch.tensor(log_probs)

    def train_rlhf(self, prompts: List[str], num_iterations: int = 100,
                   generation_kwargs: Dict = None) -> Dict[str, List[float]]:
        """
        Main RLHF training loop
        """
        if generation_kwargs is None:
            generation_kwargs = {
                'max_new_tokens': 100,
                'do_sample': True,
                'temperature': 0.7,
                'top_p': 0.9
            }

        all_metrics = []

        for iteration in range(num_iterations):
            print(f"Iteration {iteration + 1}/{num_iterations}")

            # Generate responses
            responses = self.generate_responses(prompts, generation_kwargs)

            # Compute rewards
            rewards = self.compute_rewards(prompts, responses)

            # Compute advantages (simplified)
            advantages = rewards - rewards.mean()

            # PPO update
            iteration_metrics = self.ppo_update(prompts, responses, rewards, advantages)
            all_metrics.extend(iteration_metrics)

            # Log metrics
            if (iteration + 1) % 10 == 0:
                avg_metrics = {key: np.mean([m[key] for m in all_metrics[-10:]]) for key in all_metrics[0].keys()}
                print(f"Average metrics: {avg_metrics}")

        return all_metrics
```

## 4. Evaluation and Analysis

### 4.1 Generation Quality Metrics

**Comprehensive Evaluation Framework**
```python
class GenerationEvaluator:
    """Comprehensive evaluation framework for generative models"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def evaluate_generation_quality(self, prompts: List[str], generated_texts: List[str],
                                  reference_texts: List[str] = None) -> Dict[str, float]:
        """
        Comprehensive evaluation of generation quality

        Args:
            prompts: Input prompts
            generated_texts: Generated responses
            reference_texts: Reference texts for comparison (optional)

        Returns:
            Dictionary of quality metrics
        """
        metrics = {}

        # Fluency metrics
        metrics.update(self._evaluate_fluency(generated_texts))

        # Diversity metrics
        metrics.update(self._evaluate_diversity(generated_texts))

        # Coherence metrics
        metrics.update(self._evaluate_coherence(generated_texts))

        # Relevance metrics (if references available)
        if reference_texts:
            metrics.update(self._evaluate_relevance(generated_texts, reference_texts))

        # Safety metrics
        metrics.update(self._evaluate_safety(generated_texts))

        return metrics

    def _evaluate_fluency(self, texts: List[str]) -> Dict[str, float]:
        """
        Evaluate text fluency
        """
        fluency_scores = []

        for text in texts:
            # Perplexity-based fluency (simplified)
            words = text.split()
            if len(words) > 1:
                # Simple heuristic: check for repeated words and unusual patterns
                repetition_penalty = len(set(words)) / len(words)
                avg_word_length = np.mean([len(word) for word in words])

                # Combined fluency score
                fluency = 0.7 * repetition_penalty + 0.3 * min(avg_word_length / 10, 1.0)
                fluency_scores.append(fluency)
            else:
                fluency_scores.append(0.0)

        return {
            'fluency_score': np.mean(fluency_scores),
            'fluency_std': np.std(fluency_scores)
        }

    def _evaluate_diversity(self, texts: List[str]) -> Dict[str, float]:
        """
        Evaluate text diversity
        """
        all_words = []
        unique_words = set()

        for text in texts:
            words = text.split()
            all_words.extend(words)
            unique_words.update(words)

        # Vocabulary diversity
        vocab_diversity = len(unique_words) / len(all_words) if all_words else 0

        # N-gram diversity
        bigrams = []
        for text in texts:
            words = text.split()
            bigrams.extend([(words[i], words[i+1]) for i in range(len(words)-1)])

        unique_bigrams = set(bigrams)
        bigram_diversity = len(unique_bigrams) / len(bigrams) if bigrams else 0

        return {
            'vocab_diversity': vocab_diversity,
            'bigram_diversity': bigram_diversity
        }

    def _evaluate_coherence(self, texts: List[str]) -> Dict[str, float]:
        """
        Evaluate text coherence
        """
        coherence_scores = []

        for text in texts:
            sentences = text.split('.')
            if len(sentences) > 1:
                # Simple coherence measure: sentence length variation
                sentence_lengths = [len(sent.split()) for sent in sentences if sent.strip()]
                if len(sentence_lengths) > 1:
                    length_variation = np.std(sentence_lengths)
                    normalized_variation = min(length_variation / 10, 1.0)  # Normalize
                    coherence = 1.0 - normalized_variation  # Lower variation = higher coherence
                else:
                    coherence = 1.0
            else:
                coherence = 0.5

            coherence_scores.append(coherence)

        return {
            'coherence_score': np.mean(coherence_scores),
            'coherence_std': np.std(coherence_scores)
        }

    def _evaluate_relevance(self, generated_texts: List[str], reference_texts: List[str]) -> Dict[str, float]:
        """
        Evaluate relevance to reference texts
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        # Combine all texts
        all_texts = generated_texts + reference_texts

        # Compute TF-IDF vectors
        vectorizer = TfidfVectorizer(max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(all_texts)

        # Split back
        gen_vectors = tfidf_matrix[:len(generated_texts)]
        ref_vectors = tfidf_matrix[len(generated_texts):]

        # Compute cosine similarity
        similarities = []
        for i in range(len(generated_texts)):
            sim = cosine_similarity(gen_vectors[i:i+1], ref_vectors).mean()
            similarities.append(sim)

        return {
            'relevance_score': np.mean(similarities),
            'relevance_std': np.std(similarities)
        }

    def _evaluate_safety(self, texts: List[str]) -> Dict[str, float]:
        """
        Evaluate text safety (simplified)
        """
        unsafe_patterns = [
            r'\b(hate|kill|violence|weapon|bomb)\b',
            r'\b(racial|ethnic|religious)\s+slur',
            r'\b(sexual|explicit|pornographic)\b'
        ]

        safety_scores = []

        for text in texts:
            text_lower = text.lower()
            is_safe = True

            for pattern in unsafe_patterns:
                if re.search(pattern, text_lower):
                    is_safe = False
                    break

            safety_scores.append(1.0 if is_safe else 0.0)

        return {
            'safety_score': np.mean(safety_scores),
            'unsafe_percentage': 1 - np.mean(safety_scores)
        }

    def evaluate_model_capabilities(self, model: GPT, test_prompts: List[Dict]) -> Dict[str, float]:
        """
        Evaluate specific model capabilities

        Args:
            model: Model to evaluate
            test_prompts: List of test prompts with expected capabilities

        Returns:
            Capability evaluation results
        """
        capability_scores = {}

        for capability, prompts in test_prompts.items():
            scores = []

            for prompt in prompts:
                # Generate response
                input_ids = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0)

                with torch.no_grad():
                    generated_ids = model.generate(input_ids, max_new_tokens=100, do_sample=True)
                    response = self.tokenizer.decode(generated_ids[0].tolist())

                # Evaluate response quality for capability
                score = self._evaluate_capability_response(response, capability)
                scores.append(score)

            capability_scores[capability] = {
                'score': np.mean(scores),
                'std': np.std(scores)
            }

        return capability_scores

    def _evaluate_capability_response(self, response: str, capability: str) -> float:
        """
        Evaluate response quality for specific capability
        """
        # Simplified evaluation - in practice, you'd use more sophisticated methods
        response_lower = response.lower()

        if capability == 'math_reasoning':
            # Check for mathematical content
            math_keywords = ['calculate', 'equation', 'formula', 'solve', 'math', 'number']
            return 1.0 if any(keyword in response_lower for keyword in math_keywords) else 0.0

        elif capability == 'code_generation':
            # Check for code-like content
            code_indicators = ['def ', 'class ', 'import ', 'function', 'return', 'var ']
            return 1.0 if any(indicator in response for indicator in code_indicators) else 0.0

        elif capability == 'creative_writing':
            # Check for creative content
            length_score = min(len(response) / 200, 1.0)  # Longer responses get higher scores
            return length_score

        else:
            # Generic evaluation
            return 0.5

    def benchmark_performance(self, model: GPT, benchmark_data: Dict) -> Dict[str, float]:
        """
        Benchmark model performance on standard datasets

        Args:
            model: Model to benchmark
            benchmark_data: Dictionary of benchmark datasets

        Returns:
            Benchmark results
        """
        results = {}

        for benchmark_name, data in benchmark_data.items():
            print(f"Running benchmark: {benchmark_name}")

            if benchmark_name == 'perplexity':
                # Perplexity benchmark
                perplexity = self._compute_perplexity(model, data['texts'])
                results[benchmark_name] = {'perplexity': perplexity}

            elif benchmark_name == 'generation_speed':
                # Generation speed benchmark
                speed = self._measure_generation_speed(model, data['prompts'])
                results[benchmark_name] = {'tokens_per_second': speed}

            elif benchmark_name == 'memory_usage':
                # Memory usage benchmark
                memory = self._measure_memory_usage(model, data['prompts'])
                results[benchmark_name] = {'memory_mb': memory}

        return results

    def _compute_perplexity(self, model: GPT, texts: List[str]) -> float:
        """Compute perplexity on test texts"""
        total_perplexity = 0
        num_texts = len(texts)

        for text in texts:
            input_ids = torch.tensor(self.tokenizer.encode(text), dtype=torch.long).unsqueeze(0)

            with torch.no_grad():
                outputs = model.forward(input_ids[:, :-1], targets=input_ids[:, 1:])
                loss = outputs['loss']
                perplexity = torch.exp(loss).item()

            total_perplexity += perplexity

        return total_perplexity / num_texts

    def _measure_generation_speed(self, model: GPT, prompts: List[str]) -> float:
        """Measure generation speed in tokens per second"""
        import time

        total_tokens = 0
        total_time = 0

        for prompt in prompts:
            input_ids = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0)

            start_time = time.time()
            with torch.no_grad():
                generated_ids = model.generate(input_ids, max_new_tokens=100)
            end_time = time.time()

            total_tokens += len(generated_ids[0]) - len(input_ids[0])
            total_time += end_time - start_time

        return total_tokens / total_time if total_time > 0 else 0

    def _measure_memory_usage(self, model: GPT, prompts: List[str]) -> float:
        """Measure memory usage during generation"""
        import torch.cuda as cuda

        if not cuda.is_available():
            return 0.0

        # Reset memory stats
        cuda.reset_peak_memory_stats()

        # Generate text
        for prompt in prompts[:5]:  # Use subset for memory measurement
            input_ids = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0)

            with torch.no_grad():
                model.generate(input_ids, max_new_tokens=100)

        # Get peak memory usage
        peak_memory = cuda.max_memory_allocated() / 1024 / 1024  # Convert to MB

        return peak_memory

    def generate_evaluation_report(self, metrics: Dict) -> str:
        """
        Generate comprehensive evaluation report
        """
        report = "# Generative Model Evaluation Report\n\n"

        # Summary statistics
        report += "## Summary\n"
        report += f"- Total evaluations: {len(metrics)}\n"
        report += f"- Average fluency: {metrics.get('fluency_score', 0):.3f}\n"
        report += f"- Average coherence: {metrics.get('coherence_score', 0):.3f}\n"
        report += f"- Safety score: {metrics.get('safety_score', 0):.3f}\n\n"

        # Detailed metrics
        report += "## Detailed Metrics\n"
        for metric_name, value in metrics.items():
            report += f"- {metric_name}: {value:.3f}\n"

        # Recommendations
        report += "\n## Recommendations\n"
        if metrics.get('fluency_score', 0) < 0.7:
            report += "- Consider improving model fluency through additional training\n"

        if metrics.get('diversity_score', 0) < 0.5:
            report += "- Model shows low diversity - consider increasing temperature or using nucleus sampling\n"

        if metrics.get('safety_score', 1.0) < 0.9:
            report += "- Safety concerns detected - consider adding safety filters\n"

        return report
```

## Conclusion

This comprehensive guide to transformer-based generative models provides:

1. **Core Architecture**: Complete GPT implementation with attention mechanisms
2. **Training Infrastructure**: Distributed training, optimization, and checkpointing
3. **Advanced Generation**: Multiple sampling strategies and controlled generation
4. **Efficient Fine-Tuning**: LoRA, QLoRA, adapters, and instruction tuning
5. **RLHF Implementation**: Human feedback training with PPO
6. **Evaluation Framework**: Comprehensive quality assessment and benchmarking

These implementations serve as building blocks for state-of-the-art generative AI systems and demonstrate the practical engineering considerations for deploying large language models in production environments. The code examples balance theoretical correctness with practical efficiency and can be adapted for various applications from creative writing to code generation.