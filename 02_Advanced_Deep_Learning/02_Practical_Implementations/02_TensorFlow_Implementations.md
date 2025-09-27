# TensorFlow Implementations of Advanced Deep Learning Architectures

## Overview

This section provides comprehensive TensorFlow implementations of advanced deep learning architectures, leveraging Keras API and TensorFlow 2.x features. Each implementation includes efficient training strategies, deployment considerations, and performance optimizations.

## Learning Objectives

- Implement advanced architectures using TensorFlow/Keras
- Understand TensorFlow-specific optimizations and features
- Learn distributed training and deployment strategies
- Apply implementations to production scenarios

## 1. Basic Neural Network Implementation

### 1.1 Advanced MLP with Custom Training Loop

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses, metrics
from tensorflow.keras.callbacks import Callback
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
import matplotlib.pyplot as plt
import wandb  # For experiment tracking

class CustomDense(layers.Layer):
    """
    Custom dense layer with advanced initialization and regularization.
    """
    def __init__(
        self,
        units: int,
        activation: str = 'relu',
        use_bias: bool = True,
        kernel_initializer: str = 'glorot_uniform',
        bias_initializer: str = 'zeros',
        kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
        bias_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
        activity_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias

        # Regularizers
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer

    def build(self, input_shape):
        """Build layer with custom initialization."""
        input_dim = input_shape[-1]

        # Weight initialization based on activation
        if self.activation == tf.keras.activations.relu:
            self.kernel = self.add_weight(
                name='kernel',
                shape=(input_dim, self.units),
                initializer='he_normal',
                regularizer=self.kernel_regularizer,
                trainable=True
            )
        else:
            self.kernel = self.add_weight(
                name='kernel',
                shape=(input_dim, self.units),
                initializer='glorot_uniform',
                regularizer=self.kernel_regularizer,
                trainable=True
            )

        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.units,),
                initializer='zeros',
                regularizer=self.bias_regularizer,
                trainable=True
            )

        self.built = True

    def call(self, inputs, training=None):
        """Forward pass with optional stochastic depth."""
        output = tf.matmul(inputs, self.kernel)

        if self.use_bias:
            output = output + self.bias

        if self.activation is not None:
            output = self.activation(output)

        return output

    def get_config(self):
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({
            'units': self.units,
            'activation': tf.keras.activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_regularizer': tf.keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': tf.keras.regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': tf.keras.regularizers.serialize(self.activity_regularizer),
        })
        return config

class StochasticDepth(layers.Layer):
    """
    Stochastic Depth layer for regularization.
    """
    def __init__(self, drop_rate: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.drop_rate = drop_rate

    def call(self, x, training=None):
        """Apply stochastic depth during training."""
        if training and self.drop_rate > 0:
            keep_rate = 1.0 - self.drop_rate
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
            random_tensor = tf.random.uniform(shape, dtype=x.dtype)
            keep_mask = tf.floor(keep_rate + random_tensor)
            x = x * keep_mask / keep_rate

        return x

    def get_config(self):
        config = super().get_config()
        config.update({'drop_rate': self.drop_rate})
        return config

class AdvancedMLP(keras.Model):
    """
    Advanced MLP with custom layers and training features.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str = 'relu',
        dropout_rate: float = 0.0,
        batch_norm: bool = False,
        stochastic_depth_rate: float = 0.0,
        weight_decay: float = 0.0,
        name: str = 'advanced_mlp'
    ):
        super().__init__(name=name)

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        # Create layers
        self.hidden_layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            # Regularizers
            kernel_reg = tf.keras.regularizers.l2(weight_decay) if weight_decay > 0 else None

            # Dense layer
            dense = CustomDense(
                units=hidden_dim,
                activation=activation,
                kernel_regularizer=kernel_reg,
                name=f'dense_{i+1}'
            )

            self.hidden_layers.append(dense)

            # Batch normalization
            if batch_norm:
                bn = layers.BatchNormalization(name=f'batch_norm_{i+1}')
                self.hidden_layers.append(bn)

            # Dropout
            if dropout_rate > 0:
                dropout = layers.Dropout(dropout_rate, name=f'dropout_{i+1}')
                self.hidden_layers.append(dropout)

            # Stochastic depth
            if stochastic_depth_rate > 0:
                sd_rate = stochastic_depth_rate * (i + 1) / len(hidden_dims)
                sd = StochasticDepth(sd_rate, name=f'stochastic_depth_{i+1}')
                self.hidden_layers.append(sd)

            prev_dim = hidden_dim

        # Output layer
        self.output_layer = CustomDense(
            units=output_dim,
            activation='linear',
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay) if weight_decay > 0 else None,
            name='output_layer'
        )

        # Metrics
        self.train_loss_tracker = metrics.Mean(name='train_loss')
        self.val_loss_tracker = metrics.Mean(name='val_loss')
        self.train_accuracy_tracker = metrics.CategoricalAccuracy(name='train_accuracy')
        self.val_accuracy_tracker = metrics.CategoricalAccuracy(name='val_accuracy')

    def call(self, inputs, training=None):
        """Forward pass."""
        x = inputs

        for layer in self.hidden_layers:
            x = layer(x, training=training)

        return self.output_layer(x)

    def train_step(self, data):
        """Custom training step with gradient accumulation."""
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics
        self.compiled_metrics.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        """Custom test step."""
        x, y = data
        y_pred = self(x, training=False)
        loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        self.compiled_metrics.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}

    def get_config(self):
        """Get model configuration."""
        return {
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'output_dim': self.output_dim,
        }

class CustomLearningRateScheduler(Callback):
    """
    Custom learning rate scheduler with warmup and cosine decay.
    """
    def __init__(
        self,
        max_learning_rate: float,
        warmup_steps: int,
        total_steps: int,
        min_learning_rate: float = 1e-6,
        verbose: bool = True
    ):
        super().__init__()
        self.max_learning_rate = max_learning_rate
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_learning_rate = min_learning_rate
        self.verbose = verbose
        self.current_step = 0

    def on_train_batch_begin(self, batch, logs=None):
        """Update learning rate at the beginning of each batch."""
        self.current_step += 1

        if self.current_step < self.warmup_steps:
            # Linear warmup
            learning_rate = self.max_learning_rate * (self.current_step / self.warmup_steps)
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            learning_rate = self.min_learning_rate + 0.5 * (self.max_learning_rate - self.min_learning_rate) * \
                           (1 + tf.cos(tf.constant(np.pi) * progress))

        # Update learning rate
        tf.keras.backend.set_value(self.model.optimizer.lr, learning_rate)

        if self.verbose and self.current_step % 100 == 0:
            print(f"Step {self.current_step}: Learning rate = {learning_rate:.6f}")

class WandBCallback(Callback):
    """
    Weights & Biases callback for experiment tracking.
    """
    def __init__(self, project: str, entity: Optional[str] = None, config: Optional[Dict] = None):
        super().__init__()
        self.project = project
        self.entity = entity
        self.config = config or {}

    def on_train_begin(self, logs=None):
        """Initialize W&B run."""
        if wandb.run is None:
            wandb.init(
                project=self.project,
                entity=self.entity,
                config=self.config,
                reinit=True
            )

        # Watch model
        wandb.watch(self.model, log='all', log_freq=100)

    def on_epoch_end(self, epoch, logs=None):
        """Log metrics to W&B."""
        logs = logs or {}
        wandb.log({
            'epoch': epoch,
            'train_loss': logs.get('loss', 0),
            'val_loss': logs.get('val_loss', 0),
            'train_accuracy': logs.get('categorical_accuracy', 0),
            'val_accuracy': logs.get('val_categorical_accuracy', 0),
            'learning_rate': tf.keras.backend.get_value(self.model.optimizer.lr)
        })

    def on_train_end(self, logs=None):
        """Finish W&B run."""
        wandb.finish()

# Usage Example
def advanced_mlp_example():
    """Example of advanced MLP implementation."""
    # Generate synthetic data
    np.random.seed(42)
    X = np.random.randn(10000, 20)
    y = np.random.randint(0, 5, 10000)
    y_one_hot = tf.keras.utils.to_categorical(y, num_classes=5)

    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

    # Create datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(1000).batch(128).prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_dataset = val_dataset.batch(128).prefetch(tf.data.AUTOTUNE)

    # Create model
    model = AdvancedMLP(
        input_dim=20,
        hidden_dims=[256, 128, 64],
        output_dim=5,
        activation='gelu',
        dropout_rate=0.3,
        batch_norm=True,
        stochastic_depth_rate=0.1,
        weight_decay=1e-4
    )

    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss=losses.CategoricalCrossentropy(from_logits=True),
        metrics=[metrics.CategoricalAccuracy()]
    )

    # Callbacks
    total_steps = len(train_dataset) * 100  # 100 epochs
    callbacks = [
        CustomLearningRateScheduler(
            max_learning_rate=0.001,
            warmup_steps=1000,
            total_steps=total_steps,
            verbose=True
        ),
        callbacks.EarlyStopping(
            monitor='val_categorical_accuracy',
            patience=20,
            restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6
        ),
        callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_categorical_accuracy',
            save_best_only=True,
            save_weights_only=True
        ),
        WandBCallback(
            project='advanced_mlp',
            config={
                'model_type': 'advanced_mlp',
                'hidden_dims': [256, 128, 64],
                'dropout_rate': 0.3,
                'batch_norm': True,
                'optimizer': 'adam'
            }
        )
    ]

    # Train model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=100,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate model
    val_loss, val_accuracy = model.evaluate(val_dataset, verbose=0)
    print(f"Final validation loss: {val_loss:.4f}, accuracy: {val_accuracy:.4f}")

    return model, history
```

## 2. Convolutional Neural Network Implementation

### 2.1 Efficient CNN with Custom Layers

```python
class SEBlock(layers.Layer):
    """
    Squeeze-and-Excitation block for channel attention.
    """
    def __init__(self, filters: int, reduction: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.reduction = reduction

        self.squeeze = layers.GlobalAveragePooling2D()
        self.excitation = keras.Sequential([
            layers.Dense(filters // reduction, activation='relu'),
            layers.Dense(filters, activation='sigmoid')
        ])

    def build(self, input_shape):
        """Build layer."""
        self.reshape = layers.Reshape((1, 1, self.filters))
        self.multiply = layers.Multiply()
        super().build(input_shape)

    def call(self, inputs):
        """Forward pass."""
        squeeze = self.squeeze(inputs)
        excitation = self.excitation(squeeze)
        excitation = self.reshape(excitation)
        output = self.multiply([inputs, excitation])
        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'reduction': self.reduction
        })
        return config

class GhostModule(layers.Layer):
    """
    Ghost module for efficient feature generation.
    """
    def __init__(
        self,
        filters: int,
        kernel_size: int = 1,
        ratio: int = 2,
        dw_kernel_size: int = 3,
        stride: int = 1,
        activation: str = 'relu',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.ratio = ratio
        self.dw_kernel_size = dw_kernel_size
        self.stride = stride
        self.activation = activation

        # Primary convolution
        primary_filters = filters // ratio
        self.primary_conv = keras.Sequential([
            layers.Conv2D(
                primary_filters,
                kernel_size,
                strides=stride,
                padding='same',
                use_bias=False
            ),
            layers.BatchNormalization(),
            layers.Activation(activation)
        ])

        # Depth-wise convolution for ghost features
        self.ghost_conv = keras.Sequential([
            layers.DepthwiseConv2D(
                dw_kernel_size,
                strides=1,
                padding='same',
                use_bias=False
            ),
            layers.BatchNormalization(),
            layers.Activation(activation)
        ])

    def call(self, inputs):
        """Forward pass."""
        # Primary features
        primary_features = self.primary_conv(inputs)

        # Ghost features
        ghost_features = self.ghost_conv(primary_features)

        # Concatenate features
        output = tf.concat([primary_features, ghost_features], axis=-1)

        # Ensure correct number of filters
        if output.shape[-1] > self.filters:
            output = output[:, :, :, :self.filters]

        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'ratio': self.ratio,
            'dw_kernel_size': self.dw_kernel_size,
            'stride': self.stride,
            'activation': self.activation
        })
        return config

class ConvNeXtBlock(layers.Layer):
    """
    ConvNeXt block with modern design.
    """
    def __init__(
        self,
        filters: int,
        kernel_size: int = 7,
        layer_scale: float = 1e-6,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.layer_scale = layer_scale

        self.dw_conv = layers.Conv2D(
            filters,
            kernel_size,
            padding='same',
            groups=filters,
            use_bias=False
        )

        self.pointwise_conv = layers.Conv2D(
            4 * filters,
            1,
            use_bias=False
        )

        self.act = layers.Activation('gelu')
        self.gamma = tf.Variable(layer_scale * tf.ones((4 * filters)))

        self.pointwise_conv2 = layers.Conv2D(
            filters,
            1,
            use_bias=False
        )

    def call(self, inputs):
        """Forward pass."""
        x = self.dw_conv(inputs)
        x = self.pointwise_conv(x)
        x = self.act(x)
        x = x * self.gamma
        x = self.pointwise_conv2(x)
        x = x + inputs
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'layer_scale': self.layer_scale
        })
        return config

class EfficientCNN(keras.Model):
    """
    Efficient CNN with modern components.
    """
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (32, 32, 3),
        num_classes: int = 10,
        filters_list: List[int] = [64, 128, 256, 512],
        blocks_list: List[int] = [2, 2, 6, 2],
        use_ghost: bool = True,
        use_attention: bool = True,
        dropout_rate: float = 0.5,
        name: str = 'efficient_cnn'
    ):
        super().__init__(name=name)

        self.input_shape = input_shape
        self.num_classes = num_classes

        # Stem
        self.stem = keras.Sequential([
            layers.Conv2D(filters_list[0], 3, strides=1, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.Activation('gelu')
        ])

        # Stages
        self.stages = []
        for i, (filters, num_blocks) in enumerate(zip(filters_list, blocks_list)):
            stage = keras.Sequential()

            # Downsample if needed
            if i > 0:
                stage.add(layers.Conv2D(filters, 3, strides=2, padding='same', use_bias=False))
                stage.add(layers.BatchNormalization())
                stage.add(layers.Activation('gelu'))

            # Blocks
            for j in range(num_blocks):
                if use_ghost:
                    stage.add(GhostModule(filters, activation='gelu'))
                else:
                    stage.add(ConvNeXtBlock(filters))

                if use_attention and j == num_blocks - 1:
                    stage.add(SEBlock(filters))

            self.stages.append(stage)

        # Head
        self.head = keras.Sequential([
            layers.GlobalAveragePooling2D(),
            layers.Dropout(dropout_rate),
            layers.Dense(num_classes)
        ])

    def call(self, inputs, training=None):
        """Forward pass."""
        x = inputs

        # Stem
        x = self.stem(x)

        # Stages
        for stage in self.stages:
            x = stage(x)

        # Head
        x = self.head(x)

        return x

    def get_config(self):
        return {
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
        }

def load_and_preprocess_cifar10():
    """Load and preprocess CIFAR-10 dataset."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Normalize
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Convert labels to categorical
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    # Data augmentation
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1
    )

    test_datagen = keras.preprocessing.image.ImageDataGenerator()

    train_dataset = train_datagen.flow(x_train, y_train, batch_size=128)
    test_dataset = test_datagen.flow(x_test, y_test, batch_size=128)

    return train_dataset, test_dataset

def efficient_cnn_example():
    """Example of efficient CNN implementation."""
    # Load data
    train_dataset, test_dataset = load_and_preprocess_cifar10()

    # Create model with mixed precision policy
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)

    model = EfficientCNN(
        input_shape=(32, 32, 3),
        num_classes=10,
        filters_list=[64, 128, 256, 512],
        blocks_list=[2, 2, 6, 2],
        use_ghost=True,
        use_attention=True,
        dropout_rate=0.5
    )

    # Compile with custom optimizer
    optimizer = optimizers.AdamW(
        learning_rate=1e-3,
        weight_decay=1e-4,
        clipnorm=1.0
    )

    model.compile(
        optimizer=optimizer,
        loss=losses.CategoricalCrossentropy(from_logits=True),
        metrics=[metrics.CategoricalAccuracy()]
    )

    # Callbacks
    callbacks = [
        callbacks.EarlyStopping(
            monitor='val_categorical_accuracy',
            patience=30,
            restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6
        ),
        callbacks.TensorBoard(
            log_dir='./logs',
            histogram_freq=1,
            profile_batch='500,520'
        )
    ]

    # Train model
    history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=200,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(test_dataset, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")

    return model, history
```

## 3. Transformer Implementation

### 3.1 Transformer with Custom Attention

```python
class MultiHeadAttention(layers.Layer):
    """
    Custom multi-head attention implementation.
    """
    def __init__(
        self,
        num_heads: int = 8,
        key_dim: int = 64,
        dropout: float = 0.1,
        use_causal_mask: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout = dropout
        self.use_causal_mask = use_causal_mask

    def build(self, input_shape):
        """Build attention layer."""
        # Input shape: (batch_size, seq_len, embedding_dim)
        self.embedding_dim = input_shape[-1]

        # Query, Key, Value projections
        self.query_dense = layers.Dense(self.embedding_dim)
        self.key_dense = layers.Dense(self.embedding_dim)
        self.value_dense = layers.Dense(self.embedding_dim)

        # Output projection
        self.output_dense = layers.Dense(self.embedding_dim)

        # Dropout
        self.dropout_layer = layers.Dropout(self.dropout_rate)

        super().build(input_shape)

    def call(self, inputs, mask=None, training=None):
        """Forward pass."""
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]

        # Linear projections
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        # Multi-head
        query = self._split_heads(query, batch_size)
        key = self._split_heads(key, batch_size)
        value = self._split_heads(value, batch_size)

        # Attention
        attention_output = self._scaled_dot_product_attention(
            query, key, value, mask
        )

        # Concatenate heads
        attention_output = self._combine_heads(attention_output, batch_size)

        # Final linear projection
        output = self.output_dense(attention_output)

        if training:
            output = self.dropout_layer(output)

        return output

    def _split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, key_dim)."""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.key_dim))
        return tf.transpose(x, [0, 2, 1, 3])

    def _combine_heads(self, x, batch_size):
        """Combine the attention heads."""
        x = tf.transpose(x, [0, 2, 1, 3])
        return tf.reshape(x, (batch_size, -1, self.embedding_dim))

    def _scaled_dot_product_attention(self, query, key, value, mask):
        """Compute scaled dot-product attention."""
        # Calculate attention scores
        scores = tf.matmul(query, key, transpose_b=True)
        scores = scores / tf.math.sqrt(tf.cast(self.key_dim, tf.float32))

        # Apply causal mask if needed
        if self.use_causal_mask:
            seq_len = tf.shape(scores)[-1]
            causal_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
            scores = scores - 1e9 * causal_mask

        # Apply padding mask if provided
        if mask is not None:
            scores = scores * mask

        # Attention weights
        attention_weights = tf.nn.softmax(scores, axis=-1)

        # Apply dropout to attention weights
        if training:
            attention_weights = self.dropout_layer(attention_weights)

        # Output
        output = tf.matmul(attention_weights, value)

        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_heads': self.num_heads,
            'key_dim': self.key_dim,
            'dropout': self.dropout,
            'use_causal_mask': self.use_causal_mask
        })
        return config

class TransformerEncoder(layers.Layer):
    """
    Transformer encoder layer.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout

        # Self-attention
        self.attention = MultiHeadAttention(num_heads, embed_dim, dropout)

        # Feed-forward network
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(embed_dim),
            layers.Dropout(dropout)
        ])

        # Normalization layers
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        # Dropout
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, inputs, mask=None, training=None):
        """Forward pass."""
        # Self-attention
        attn_output = self.attention(inputs, mask=mask, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        # Feed-forward
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        output = self.layernorm2(out1 + ffn_output)

        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'dropout': self.dropout
        })
        return config

class PositionalEncoding(layers.Layer):
    """
    Sinusoidal positional encoding.
    """
    def __init__(self, max_position_embeddings: int = 512, **kwargs):
        super().__init__(**kwargs)
        self.max_position_embeddings = max_position_embeddings

    def build(self, input_shape):
        """Build positional encoding."""
        self.position_encoding = self._create_position_encoding(
            input_shape[-1],
            self.max_position_embeddings
        )
        super().build(input_shape)

    def call(self, inputs):
        """Add positional encoding to inputs."""
        seq_len = tf.shape(inputs)[1]
        return inputs + self.position_encoding[:, :seq_len, :]

    def _create_position_encoding(self, embed_dim, max_len):
        """Create sinusoidal positional encoding."""
        position = tf.range(max_len, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(
            tf.range(0, embed_dim, 2, dtype=tf.float32) *
            (-tf.math.log(10000.0) / embed_dim)
        )

        pe = tf.zeros((max_len, embed_dim), dtype=tf.float32)
        pe = tf.tensor_scatter_nd_update(
            pe,
            tf.stack([tf.range(max_len), tf.range(0, embed_dim, 2)], axis=1),
            tf.sin(position * div_term)
        )
        pe = tf.tensor_scatter_nd_update(
            pe,
            tf.stack([tf.range(max_len), tf.range(1, embed_dim, 2)], axis=1),
            tf.cos(position * div_term)
        )

        return pe[tf.newaxis, :, :]

    def get_config(self):
        config = super().get_config()
        config.update({'max_position_embeddings': self.max_position_embeddings})
        return config

class Transformer(keras.Model):
    """
    Complete Transformer model.
    """
    def __init__(
        self,
        vocab_size: int,
        max_length: int,
        embed_dim: int = 512,
        num_heads: int = 8,
        ff_dim: int = 2048,
        num_layers: int = 6,
        dropout: float = 0.1,
        name: str = 'transformer'
    ):
        super().__init__(name=name)

        self.vocab_size = vocab_size
        self.max_length = max_length

        # Token and position embeddings
        self.token_embedding = layers.Embedding(vocab_size, embed_dim)
        self.position_embedding = PositionalEncoding(max_length)

        # Encoder layers
        self.encoder_layers = [
            TransformerEncoder(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ]

        # Dropout
        self.dropout = layers.Dropout(dropout)

        # Output layer
        self.output_layer = layers.Dense(vocab_size)

    def call(self, inputs, mask=None, training=None):
        """Forward pass."""
        # Token embeddings
        x = self.token_embedding(inputs)

        # Positional encoding
        x = self.position_embedding(x)

        # Dropout
        x = self.dropout(x, training=training)

        # Encoder layers
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, mask=mask, training=training)

        # Output
        output = self.output_layer(x)

        return output

    def create_padding_mask(self, x):
        """Create padding mask."""
        mask = tf.cast(tf.math.not_equal(x, 0), tf.float32)
        return mask[:, tf.newaxis, tf.newaxis, :]

    def get_config(self):
        return {
            'vocab_size': self.vocab_size,
            'max_length': self.max_length,
        }

# Custom training loop for language modeling
class TransformerTrainer:
    """
    Custom trainer for transformer with distributed support.
    """
    def __init__(
        self,
        model: Transformer,
        optimizer: optimizers.Optimizer,
        loss_fn: losses.Loss,
        metrics: List[metrics.Metric],
        strategy: Optional[tf.distribute.Strategy] = None
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.strategy = strategy or tf.distribute.get_strategy()

    @tf.function
    def train_step(self, batch):
        """Training step with gradient accumulation."""
        inputs, targets = batch

        with tf.GradientTape() as tape:
            predictions = self.model(inputs, training=True)
            loss = self.loss_fn(targets, predictions)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # Update metrics
        for metric in self.metrics:
            metric.update_state(targets, predictions)

        return {'loss': loss}

    @tf.function
    def test_step(self, batch):
        """Test step."""
        inputs, targets = batch
        predictions = self.model(inputs, training=False)
        loss = self.loss_fn(targets, predictions)

        # Update metrics
        for metric in self.metrics:
            metric.update_state(targets, predictions)

        return {'loss': loss}

    def train(self, train_dataset, epochs: int, val_dataset=None):
        """Training loop."""
        with self.strategy.scope():
            for epoch in range(epochs):
                print(f"Epoch {epoch + 1}/{epochs}")

                # Training
                train_loss = 0.0
                for batch in train_dataset:
                    step_metrics = self.train_step(batch)
                    train_loss += step_metrics['loss']

                train_loss /= len(train_dataset)

                # Validation
                if val_dataset:
                    val_loss = 0.0
                    for batch in val_dataset:
                        step_metrics = self.test_step(batch)
                        val_loss += step_metrics['loss']
                    val_loss /= len(val_dataset)

                    # Print metrics
                    metric_values = {m.name: m.result().numpy() for m in self.metrics}
                    print(f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")
                    print(f"Metrics: {metric_values}")
                else:
                    print(f"Train loss: {train_loss:.4f}")

                # Reset metrics
                for metric in self.metrics:
                    metric.reset_states()

def transformer_example():
    """Example of transformer implementation."""
    # Create dummy data
    vocab_size = 10000
    max_length = 128
    batch_size = 32

    # Generate synthetic sequences
    train_data = tf.random.uniform((1000, max_length), maxval=vocab_size, dtype=tf.int32)
    train_labels = tf.random.uniform((1000, max_length), maxval=vocab_size, dtype=tf.int32)

    # Create datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Create model
    model = Transformer(
        vocab_size=vocab_size,
        max_length=max_length,
        embed_dim=512,
        num_heads=8,
        ff_dim=2048,
        num_layers=6,
        dropout=0.1
    )

    # Setup training
    optimizer = optimizers.Adam(learning_rate=1e-4)
    loss_fn = losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = [metrics.SparseCategoricalAccuracy()]

    # Create trainer
    trainer = TransformerTrainer(model, optimizer, loss_fn, metrics)

    # Train
    trainer.train(train_dataset, epochs=10)

    return model
```

## 4. Distributed Training and Deployment

### 4.1 Multi-GPU and TPU Training

```python
def setup_multi_gpu_training():
    """Setup multi-GPU training strategy."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be set before GPUs are initialized
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)

    # Create MirroredStrategy
    strategy = tf.distribute.MirroredStrategy()
    print(f"Number of devices: {strategy.num_replicas_in_sync}")

    return strategy

def setup_tpu_training():
    """Setup TPU training strategy."""
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
        print(f"TPU devices: {strategy.num_replicas_in_sync}")
        return strategy
    except ValueError:
        print("TPU not found, using default strategy")
        return tf.distribute.get_strategy()

def create_distributed_model(strategy, input_shape, num_classes):
    """Create model within distribution strategy."""
    with strategy.scope():
        # Create model
        model = EfficientCNN(
            input_shape=input_shape,
            num_classes=num_classes,
            filters_list=[64, 128, 256, 512],
            blocks_list=[2, 2, 6, 2]
        )

        # Compile model
        optimizer = optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4)
        model.compile(
            optimizer=optimizer,
            loss=losses.CategoricalCrossentropy(from_logits=True),
            metrics=[metrics.CategoricalAccuracy()]
        )

        return model

def distributed_training_example():
    """Example of distributed training."""
    # Setup strategy
    strategy = setup_multi_gpu_training()  # or setup_tpu_training()

    # Create distributed datasets
    batch_size_per_replica = 32
    global_batch_size = batch_size_per_replica * strategy.num_replicas_in_sync

    # Load data
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    # Create distributed datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(10000).batch(global_batch_size)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(global_batch_size)
    test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

    # Create distributed model
    model = create_distributed_model(strategy, (32, 32, 3), 10)

    # Train
    model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=100,
        callbacks=[
            callbacks.EarlyStopping(patience=20, restore_best_weights=True),
            callbacks.TensorBoard(log_dir='./distributed_logs')
        ]
    )

    return model

# Model Quantization and Optimization
def quantize_model(model):
    """Quantize model for deployment."""
    # Convert to TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter.allow_custom_ops = True

    # Convert
    tflite_model = converter.convert()

    # Save
    with open('quantized_model.tflite', 'wb') as f:
        f.write(tflite_model)

    return tflite_model

def prune_model(model):
    """Prune model for compression."""
    import tensorflow_model_optimization as tfmot

    # Define pruning schedule
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(
            0.5, begin_step=0, frequency=100
        )
    }

    # Apply pruning
    pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)

    # Recompile
    pruned_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return pruned_model

# Serving and Deployment
def export_for_serving(model):
    """Export model for serving."""
    # Save model in SavedModel format
    tf.saved_model.save(model, 'saved_model/1/')

    # Convert to TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_saved_model('saved_model/1/')
    tflite_model = converter.convert()

    # Save TFLite model
    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)

    print("Model exported for serving")

def create_inference_function(model):
    """Create inference function for deployment."""
    @tf.function
    def predict_fn(inputs):
        return model(inputs, training=False)

    return predict_fn

# Benchmark models
def benchmark_models(models_dict, test_data):
    """Benchmark multiple models."""
    results = {}

    for name, model in models_dict.items():
        # Warmup
        for _ in range(10):
            model.predict(test_data[:1], verbose=0)

        # Benchmark
        import time
        start_time = time.time()
        predictions = model.predict(test_data, verbose=0)
        end_time = time.time()

        results[name] = {
            'inference_time': end_time - start_time,
            'throughput': len(test_data) / (end_time - start_time),
            'model_size': sum([tf.reduce_prod(var.shape).numpy() for var in model.trainable_variables]) * 4 / 1024 / 1024  # MB
        }

    return results

# Usage Example
def tensorflow_advanced_example():
    """Comprehensive example of advanced TensorFlow features."""
    print("=== TensorFlow Advanced Implementation Example ===")

    # Example 1: Advanced MLP
    print("\n1. Advanced MLP Example")
    mlp_model, mlp_history = advanced_mlp_example()

    # Example 2: Efficient CNN
    print("\n2. Efficient CNN Example")
    cnn_model, cnn_history = efficient_cnn_example()

    # Example 3: Transformer
    print("\n3. Transformer Example")
    transformer_model = transformer_example()

    # Example 4: Distributed Training
    print("\n4. Distributed Training Example")
    try:
        distributed_model = distributed_training_example()
    except Exception as e:
        print(f"Distributed training failed: {e}")

    # Example 5: Model Optimization
    print("\n5. Model Optimization")
    # Quantize CNN model
    tflite_model = quantize_model(cnn_model)
    print(f"Quantized model size: {len(tflite_model) / 1024 / 1024:.2f} MB")

    return {
        'mlp_model': mlp_model,
        'cnn_model': cnn_model,
        'transformer_model': transformer_model,
        'tflite_model': tflite_model
    }

if __name__ == "__main__":
    results = tensorflow_advanced_example()
    print("\nAll examples completed successfully!")
```

## Summary

This comprehensive TensorFlow implementation guide covers:

1. **Advanced MLP** with custom layers, training loops, and experiment tracking
2. **Efficient CNN** with modern components like Ghost modules and ConvNeXt blocks
3. **Transformer** with custom attention and distributed training support
4. **Distributed Training** with multi-GPU and TPU strategies
5. **Model Optimization** including quantization, pruning, and deployment

Key features include:
- Custom layer implementations with proper serialization
- Mixed precision training for performance
- Distributed training with automatic data parallelism
- Model optimization and deployment tools
- Comprehensive logging and monitoring
- Efficient data pipelines and preprocessing

These implementations demonstrate best practices for building production-ready deep learning models in TensorFlow.

## Key References

- Abadi, M., et al. (2016). TensorFlow: A System for Large-Scale Machine Learning.
- Chollet, F. (2021). Deep Learning with Python (2nd ed.).
- Liu, Z., et al. (2022). A ConvNet for the 2020s. CVPR.
- Vaswani, A., et al. (2017). Attention is all you need. NeurIPS.

## Exercises

1. Implement a custom activation function and integrate it into the MLP
2. Add gradient checkpointing to reduce memory usage in the transformer
3. Implement knowledge distillation between teacher and student models
4. Create a custom callback for advanced monitoring and early stopping
5. Build a model ensemble and deploy it as a single TensorFlow Lite model