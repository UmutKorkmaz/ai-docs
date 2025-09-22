# Deep Learning Practical Implementations

## Production-Ready Examples and Best Practices

---

## 1. Complete Training Pipeline

### Data Loading and Preprocessing

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time
import os
from pathlib import Path

class CustomDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.targets[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, label

def get_data_loaders(data_dir, batch_size=32, img_size=224):
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Validation transform (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Example dataset creation (replace with your actual data loading)
    train_data = torch.randn(1000, 3, 224, 224)
    train_labels = torch.randint(0, 10, (1000,))
    val_data = torch.randn(200, 3, 224, 224)
    val_labels = torch.randint(0, 10, (200,))

    train_dataset = CustomDataset(train_data, train_labels, train_transform)
    val_dataset = CustomDataset(val_data, val_labels, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader
```

### Model Architecture with Modern Techniques

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout_rate)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)

class ModernCNN(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super().__init__()

        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(3, 2, 1)

        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2, 1)
        self.layer2 = self._make_layer(64, 128, 2, 2)
        self.layer3 = self._make_layer(128, 256, 2, 2)
        self.layer4 = self._make_layer(256, 512, 2, 2)

        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)
        self.dropout = nn.Dropout2d(dropout_rate)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        x = self.se(x)
        x += residual
        return F.relu(x)
```

### Advanced Training with Mixed Precision and Gradient Clipping

```python
import torch.cuda.amp as amp

class AdvancedTrainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # Set up device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Mixed precision setup
        self.use_amp = config.get('use_amp', True) and torch.cuda.is_available()
        if self.use_amp:
            self.scaler = amp.GradScaler()

        # Optimizer and scheduler
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()

        # Loss function
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.get('label_smoothing', 0.1))

        # Logging
        self.writer = SummaryWriter(config['log_dir'])
        self.best_val_acc = 0.0

        # Create checkpoint directory
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _get_optimizer(self):
        optimizer_config = self.config.get('optimizer', {})
        optimizer_type = optimizer_config.get('type', 'adamw')

        if optimizer_type == 'adamw':
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=optimizer_config.get('lr', 1e-3),
                weight_decay=optimizer_config.get('weight_decay', 1e-4),
                betas=optimizer_config.get('betas', (0.9, 0.999))
            )
        elif optimizer_type == 'sgd':
            return torch.optim.SGD(
                self.model.parameters(),
                lr=optimizer_config.get('lr', 1e-2),
                momentum=optimizer_config.get('momentum', 0.9),
                weight_decay=optimizer_config.get('weight_decay', 1e-4)
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    def _get_scheduler(self):
        scheduler_config = self.config.get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'cosine')

        if scheduler_type == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['epochs'],
                eta_min=scheduler_config.get('min_lr', 1e-6)
            )
        elif scheduler_type == 'reduce_on_plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 10),
                verbose=True
            )
        elif scheduler_type == 'one_cycle':
            steps_per_epoch = len(self.train_loader)
            return torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=scheduler_config.get('max_lr', 1e-3),
                epochs=self.config['epochs'],
                steps_per_epoch=steps_per_epoch
            )
        else:
            return None

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        progress_bar = enumerate(self.train_loader)
        for batch_idx, (data, target) in progress_bar:
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()

            # Mixed precision forward pass
            if self.use_amp:
                with amp.autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)

                # Scale gradients
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.config.get('grad_clip', 0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['grad_clip']
                    )

                # Update weights
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()

                # Gradient clipping
                if self.config.get('grad_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['grad_clip']
                    )

                self.optimizer.step()

            # Update learning rate
            if self.scheduler and isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                self.scheduler.step()

            # Metrics
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            # Logging
            if batch_idx % self.config.get('log_interval', 100) == 0:
                acc = 100. * correct / total
                avg_loss = total_loss / (batch_idx + 1)
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {avg_loss:.4f}, Acc: {acc:.2f}%')

        # Update learning rate for other schedulers
        if self.scheduler and not isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
            self.scheduler.step()

        return total_loss / len(self.train_loader), 100. * correct / total

    def validate(self, epoch):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_targets = []
        all_predictions = []

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)

                if self.use_amp:
                    with amp.autocast():
                        output = self.model(data)
                        loss = self.criterion(output, target)
                else:
                    output = self.model(data)
                    loss = self.criterion(output, target)

                val_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

                all_targets.extend(target.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        # Calculate metrics
        val_acc = 100. * correct / total
        val_loss = val_loss / len(self.val_loader)

        # Additional metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average='weighted'
        )

        print(f'Validation - Epoch {epoch}: Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, '
              f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')

        # Log to tensorboard
        self.writer.add_scalar('val/loss', val_loss, epoch)
        self.writer.add_scalar('val/accuracy', val_acc, epoch)
        self.writer.add_scalar('val/precision', precision, epoch)
        self.writer.add_scalar('val/recall', recall, epoch)
        self.writer.add_scalar('val/f1', f1, epoch)

        # Learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('learning_rate', current_lr, epoch)

        # Update scheduler if it's ReduceLROnPlateau
        if self.scheduler and isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(val_acc)

        return val_loss, val_acc, precision, recall, f1

    def train(self):
        print(f"Starting training on {self.device}")
        print(f"Using mixed precision: {self.use_amp}")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(1, self.config['epochs'] + 1):
            start_time = time.time()

            # Training
            train_loss, train_acc = self.train_epoch(epoch)

            # Validation
            val_loss, val_acc, precision, recall, f1 = self.validate(epoch)

            # Logging
            epoch_time = time.time() - start_time
            print(f'Epoch {epoch} completed in {epoch_time:.2f}s')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

            # Log to tensorboard
            self.writer.add_scalar('train/loss', train_loss, epoch)
            self.writer.add_scalar('train/accuracy', train_acc, epoch)
            self.writer.add_scalar('epoch_time', epoch_time, epoch)

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(epoch, val_acc, is_best=True)

            # Save regular checkpoint
            if epoch % self.config.get('save_interval', 10) == 0:
                self.save_checkpoint(epoch, val_acc, is_best=False)

        print(f"Training completed. Best validation accuracy: {self.best_val_acc:.2f}%")
        self.writer.close()

    def save_checkpoint(self, epoch, val_acc, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'config': self.config
        }

        if self.use_amp:
            checkpoint['scaler'] = self.scaler.state_dict()

        if self.scheduler:
            checkpoint['scheduler'] = self.scheduler.state_dict()

        filename = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, filename)

        if is_best:
            best_filename = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_filename)
            print(f"New best model saved with accuracy: {val_acc:.2f}%")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        if self.use_amp and 'scaler' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler'])

        if self.scheduler and 'scheduler' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler'])

        print(f"Checkpoint loaded from {checkpoint_path}")
        return checkpoint['epoch'], checkpoint['val_acc']
```

### Configuration and Training Script

```python
def get_training_config():
    return {
        'epochs': 100,
        'batch_size': 64,
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'grad_clip': 1.0,
        'use_amp': True,
        'label_smoothing': 0.1,
        'dropout_rate': 0.5,

        'optimizer': {
            'type': 'adamw',
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'betas': (0.9, 0.999)
        },

        'scheduler': {
            'type': 'cosine',
            'min_lr': 1e-6
        },

        'data': {
            'img_size': 224,
            'augmentation': True
        },

        'log_interval': 100,
        'save_interval': 10,
        'log_dir': './logs',
        'checkpoint_dir': './checkpoints'
    }

def main():
    # Configuration
    config = get_training_config()

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    # Data loaders
    train_loader, val_loader = get_data_loaders(
        './data',
        batch_size=config['batch_size'],
        img_size=config['data']['img_size']
    )

    # Model
    model = ModernCNN(num_classes=10, dropout_rate=config['dropout_rate'])

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Trainer
    trainer = AdvancedTrainer(model, train_loader, val_loader, config)

    # Train
    trainer.train()

if __name__ == '__main__':
    main()
```

---

## 2. Transfer Learning with Hugging Face

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from datasets import load_dataset
import evaluate
import numpy as np
import torch

class TransferLearningNLP:
    def __init__(self, model_name, num_labels, max_length=512):
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )

        # Data collator
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        # Metrics
        self.accuracy_metric = evaluate.load("accuracy")
        self.f1_metric = evaluate.load("f1")

    def preprocess_function(self, examples):
        return self.tokenizer(
            examples['text'],
            truncation=True,
            padding=False,  # Let the collator handle padding
            max_length=self.max_length
        )

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        accuracy = self.accuracy_metric.compute(predictions=predictions, references=labels)
        f1 = self.f1_metric.compute(predictions=predictions, references=labels, average='weighted')

        return {
            'accuracy': accuracy['accuracy'],
            'f1': f1['f1']
        }

    def load_and_prepare_data(self, dataset_name, text_column='text', label_column='label'):
        # Load dataset
        dataset = load_dataset(dataset_name)

        # Preprocess
        processed_dataset = dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=dataset['train'].column_names
        )

        # Rename label column if needed
        if label_column in ['train', 'test', 'validation']:
            processed_dataset = processed_dataset.rename_column('label', 'labels')

        return processed_dataset

    def train_model(self, dataset, output_dir='./results', num_epochs=3, batch_size=16):
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=100,
            evaluation_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model='f1',
            greater_is_better=True,
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
        )

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['validation'],
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )

        # Train
        trainer.train()

        # Save model
        trainer.save_model(f'{output_dir}/final_model')

        return trainer

class LoRATrainer:
    def __init__(self, base_model_name, num_labels, rank=8, alpha=16):
        self.model_name = base_model_name
        self.num_labels = num_labels
        self.rank = rank
        self.alpha = alpha

        # Load base model
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=num_labels
        )

        # Apply LoRA
        self.model = self._apply_lora(self.base_model)

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    def _apply_lora(self, model):
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and 'classifier' not in name:
                # Create LoRA layer
                in_features = module.in_features
                out_features = module.out_features

                # Replace with LoRA layer
                lora_layer = LoRALayer(module, self.rank, self.alpha)

                # Replace in model
                parent = model
                name_parts = name.split('.')
                for part in name_parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, name_parts[-1], lora_layer)

        return model

class LoRALayer(torch.nn.Module):
    def __init__(self, original_layer, rank, alpha):
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

        self.lora_A = torch.nn.Parameter(torch.randn(rank, in_features))
        self.lora_B = torch.nn.Parameter(torch.randn(out_features, rank))

        # Scaling
        self.scaling = alpha / rank

        # Initialize
        torch.nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        torch.nn.init.zeros_(self.lora_B)

    def forward(self, x):
        original_output = self.original_layer(x)
        lora_output = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return original_output + lora_output
```

---

## 3. Computer Vision Implementation

```python
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

class SegmentationTrainer:
    def __init__(self, model_name, num_classes, input_size=(512, 512)):
        self.input_size = input_size
        self.num_classes = num_classes

        # Create model
        self.model = smp.create_model(
            model_name,
            encoder_name='resnet50',
            encoder_weights='imagenet',
            classes=num_classes,
            activation='softmax2d' if num_classes > 1 else 'sigmoid'
        )

        # Loss function
        if num_classes > 1:
            self.criterion = smp.losses.DiceLoss(mode='multiclass')
        else:
            self.criterion = smp.losses.DiceLoss(mode='binary')

        # Metrics
        self.metrics = [
            smp.utils.metrics.IoU(threshold=0.5),
            smp.utils.metrics.Fscore(beta=1, threshold=0.5),
            smp.utils.metrics.Accuracy(threshold=0.5),
            smp.utils.metrics.Recall(threshold=0.5),
            smp.utils.metrics.Precision(threshold=0.5),
        ]

    def get_train_transform(self):
        return A.Compose([
            A.Resize(*self.input_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=45, p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.GaussNoise(p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    def get_val_transform(self):
        return A.Compose([
            A.Resize(*self.input_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    def get_training_config(self):
        return {
            'optimizer': torch.optim.AdamW(self.model.parameters(), lr=1e-4),
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(
                torch.optim.AdamW(self.model.parameters(), lr=1e-4),
                T_max=100
            ),
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'epochs': 100,
            'batch_size': 8,
        }

class ObjectDetectionTrainer:
    def __init__(self, num_classes, pretrained=True):
        self.num_classes = num_classes

        # Load pre-trained model
        if pretrained:
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
            # Replace classifier head
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        else:
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, num_classes=num_classes)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def get_transform(self, train=True):
        transforms = []
        transforms.append(T.ToTensor())
        if train:
            transforms.append(T.RandomHorizontalFlip(0.5))
        return T.Compose(transforms)

    def collate_fn(self, batch):
        return tuple(zip(*batch))

    def train_one_epoch(self, train_loader, optimizer):
        self.model.train()
        total_loss = 0

        for images, targets in train_loader:
            images = [image.to(self.device) for image in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            losses.backward()
            optimizer.step()

            total_loss += losses.item()

        return total_loss / len(train_loader)

    def evaluate(self, val_loader):
        self.model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for images, targets in val_loader:
                images = [image.to(self.device) for image in images]
                predictions = self.model(images)

                all_predictions.extend(predictions)
                all_targets.extend(targets)

        # Calculate mAP
        return self.calculate_map(all_predictions, all_targets)
```

---

## 4. Deployment and Inference

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Any, List
import json
import time
from abc import ABC, abstractmethod

class ModelDeployer(ABC):
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.model_path = model_path
        self.device = torch.device(device)
        self.model = self.load_model()
        self.preprocessor = self.get_preprocessor()
        self.postprocessor = self.get_postprocessor()

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def get_preprocessor(self):
        pass

    @abstractmethod
    def get_postprocessor(self):
        pass

    @abstractmethod
    def predict(self, input_data: Any) -> Any:
        pass

    def batch_predict(self, input_data: List[Any], batch_size: int = 32) -> List[Any]:
        results = []
        for i in range(0, len(input_data), batch_size):
            batch = input_data[i:i + batch_size]
            batch_results = self.predict(batch)
            results.extend(batch_results)
        return results

class PyTorchDeployer(ModelDeployer):
    def __init__(self, model_path: str, model_class: nn.Module = None, model_config: Dict = None, **kwargs):
        self.model_class = model_class
        self.model_config = model_config or {}
        super().__init__(model_path, **kwargs)

    def load_model(self):
        checkpoint = torch.load(self.model_path, map_location=self.device)

        if self.model_class:
            model = self.model_class(**self.model_config)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model = checkpoint['model']

        model.to(self.device)
        model.eval()
        return model

    def get_preprocessor(self):
        return nn.Identity()

    def get_postprocessor(self):
        return nn.Identity()

    def predict(self, input_data):
        if isinstance(input_data, list):
            # Batch prediction
            input_tensor = torch.stack([self.preprocessor(item) for item in input_data])
        else:
            # Single prediction
            input_tensor = self.preprocessor(input_data).unsqueeze(0)

        input_tensor = input_tensor.to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)

        output = self.postprocessor(output)

        if isinstance(input_data, list):
            return output.cpu().numpy()
        else:
            return output.cpu().numpy()[0]

class ONNXDeployer(ModelDeployer):
    def __init__(self, model_path: str, **kwargs):
        import onnxruntime as ort
        super().__init__(model_path, **kwargs)
        self.ort_session = ort.InferenceSession(model_path)

    def load_model(self):
        # ONNX models are loaded in the inference session
        return None

    def predict(self, input_data):
        # Convert input to ONNX format
        input_name = self.ort_session.get_inputs()[0].name
        if isinstance(input_data, list):
            input_tensor = np.stack([self.preprocessor(item) for item in input_data])
        else:
            input_tensor = self.preprocessor(input_data).unsqueeze(0)

        # Run inference
        outputs = self.ort_session.run(None, {input_name: input_tensor})

        # Post-process
        return self.postprocessor(outputs[0])

class TritonInferenceClient:
    def __init__(self, server_url: str, model_name: str, model_version: str = '1'):
        import tritonclient.http as httpclient
        self.server_url = server_url
        self.model_name = model_name
        self.model_version = model_version
        self.client = httpclient.InferenceServerClient(url=server_url)

    def predict(self, input_data: np.ndarray):
        # Create input tensor
        inputs = [httpclient.InferInput('input__0', input_data.shape, 'FP32')]
        inputs[0].set_data_from_numpy(input_data)

        # Make prediction
        outputs = [httpclient.InferRequestedOutput('output__0')]
        result = self.client.infer(self.model_name, inputs, outputs=outputs, model_version=self.model_version)

        return result.as_numpy('output__0')

class ModelBenchmark:
    def __init__(self, model_deployer: ModelDeployer, warmup_iterations: int = 10):
        self.model_deployer = model_deployer
        self.warmup_iterations = warmup_iterations

    def warmup(self):
        print("Warming up model...")
        dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        for _ in range(self.warmup_iterations):
            _ = self.model_deployer.predict(dummy_input)
        print("Warmup complete.")

    def benchmark_latency(self, input_data: np.ndarray, num_iterations: int = 100):
        print(f"Benchmarking latency with {num_iterations} iterations...")

        # Warmup
        self.warmup()

        # Measure latency
        latencies = []
        for _ in range(num_iterations):
            start_time = time.time()
            _ = self.model_deployer.predict(input_data)
            end_time = time.time()
            latencies.append((end_time - start_time) * 1000)  # Convert to ms

        # Calculate statistics
        avg_latency = np.mean(latencies)
        p50_latency = np.percentile(latencies, 50)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)

        print(f"Average latency: {avg_latency:.2f} ms")
        print(f"P50 latency: {p50_latency:.2f} ms")
        print(f"P95 latency: {p95_latency:.2f} ms")
        print(f"P99 latency: {p99_latency:.2f} ms")

        return {
            'avg_latency': avg_latency,
            'p50_latency': p50_latency,
            'p95_latency': p95_latency,
            'p99_latency': p99_latency
        }

    def benchmark_throughput(self, input_data: np.ndarray, duration_seconds: int = 60):
        print(f"Benchmarking throughput for {duration_seconds} seconds...")

        # Warmup
        self.warmup()

        # Measure throughput
        start_time = time.time()
        end_time = start_time + duration_seconds
        num_predictions = 0

        while time.time() < end_time:
            _ = self.model_deployer.predict(input_data)
            num_predictions += 1

        throughput = num_predictions / duration_seconds
        print(f"Throughput: {throughput:.2f} predictions/second")

        return throughput

class ModelExporter:
    @staticmethod
    def export_to_onnx(model: nn.Module, input_shape: tuple, output_path: str):
        # Create dummy input
        dummy_input = torch.randn(input_shape)

        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )

        print(f"Model exported to {output_path}")

    @staticmethod
    def export_to_torchscript(model: nn.Module, output_path: str):
        # Convert to TorchScript
        scripted_model = torch.jit.script(model)
        scripted_model.save(output_path)

        print(f"Model exported to {output_path}")

    @staticmethod
    def export_to_tensorrt(model: nn.Module, input_shape: tuple, output_path: str):
        try:
            import tensorrt as trt
            import torch_tensorrt

            # Convert to TensorRT
            trt_model = torch_tensorrt.compile(
                model,
                inputs=[torch_tensorrt.Input(input_shape)],
                enabled_precisions={torch.float, torch.half}
            )

            # Save TensorRT engine
            with open(output_path, 'wb') as f:
                f.write(trt_model.engine.serialize())

            print(f"Model exported to {output_path}")
        except ImportError:
            print("TensorRT not available. Install torch-tensorrt to enable TensorRT export.")

# Usage examples
if __name__ == "__main__":
    # Example: Deploy a trained model
    model_path = './checkpoints/best_model.pth'
    model_class = ModernCNN
    model_config = {'num_classes': 10}

    # Create deployer
    deployer = PyTorchDeployer(
        model_path=model_path,
        model_class=model_class,
        model_config=model_config,
        device='cuda'
    )

    # Benchmark performance
    benchmark = ModelBenchmark(deployer)
    dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)

    latency_stats = benchmark.benchmark_latency(dummy_input)
    throughput = benchmark.benchmark_throughput(dummy_input)

    # Export to different formats
    model = deployer.model
    ModelExporter.export_to_onnx(model, (1, 3, 224, 224), 'model.onnx')
    ModelExporter.export_to_torchscript(model, 'model.pt')
```

---

## 5. Monitoring and Experiment Tracking

```python
import torch
import numpy as np
from typing import Dict, Any, List
import wandb
import mlflow
import mlflow.pytorch
from pathlib import Path
import json
import time
from datetime import datetime

class ExperimentTracker:
    def __init__(self, experiment_name: str, config: Dict[str, Any]):
        self.experiment_name = experiment_name
        self.config = config

        # Initialize trackers
        self._init_wandb()
        self._init_mlflow()

        # Metrics storage
        self.metrics_history = {}
        self.best_metrics = {}

    def _init_wandb(self):
        if self.config.get('use_wandb', False):
            wandb.init(
                project=self.experiment_name,
                config=self.config,
                name=self.config.get('run_name', f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            )

    def _init_mlflow(self):
        if self.config.get('use_mlflow', False):
            mlflow.set_experiment(self.experiment_name)
            mlflow.start_run(run_name=self.config.get('run_name'))
            mlflow.log_params(self.config)

    def log_metrics(self, metrics: Dict[str, float], step: int):
        # Log to wandb
        if self.config.get('use_wandb', False):
            wandb.log(metrics, step=step)

        # Log to mlflow
        if self.config.get('use_mlflow', False):
            mlflow.log_metrics(metrics, step=step)

        # Store locally
        for key, value in metrics.items():
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            self.metrics_history[key].append((step, value))

            # Update best metrics
            if key not in self.best_metrics or value > self.best_metrics[key]:
                self.best_metrics[key] = value

    def log_model(self, model: torch.nn.Module, artifact_path: str):
        if self.config.get('use_mlflow', False):
            mlflow.pytorch.log_model(model, artifact_path)

    def log_artifact(self, file_path: str):
        if self.config.get('use_wandb', False):
            wandb.save(file_path)

        if self.config.get('use_mlflow', False):
            mlflow.log_artifact(file_path)

    def finish(self):
        if self.config.get('use_wandb', False):
            wandb.finish()

        if self.config.get('use_mlflow', False):
            mlflow.end_run()

class ModelMonitor:
    def __init__(self, model: torch.nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.hooks = []

    def monitor_activations(self, layer_names: List[str] = None):
        activations = {}

        def hook_fn(name):
            def hook(module, input, output):
                activations[name] = {
                    'mean': output.mean().item(),
                    'std': output.std().item(),
                    'min': output.min().item(),
                    'max': output.max().item(),
                    'shape': list(output.shape)
                }
            return hook

        if layer_names is None:
            # Monitor all layers
            for name, module in self.model.named_modules():
                if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.ReLU)):
                    hook = module.register_forward_hook(hook_fn(name))
                    self.hooks.append(hook)
        else:
            # Monitor specific layers
            for name, module in self.model.named_modules():
                if name in layer_names:
                    hook = module.register_forward_hook(hook_fn(name))
                    self.hooks.append(hook)

        return activations

    def monitor_gradients(self):
        grad_stats = {}

        def hook_fn(name):
            def hook(module, grad_input, grad_output):
                if grad_output[0] is not None:
                    grad = grad_output[0]
                    grad_stats[name] = {
                        'mean': grad.mean().item(),
                        'std': grad.std().item(),
                        'min': grad.min().item(),
                        'max': grad.max().item(),
                        'norm': grad.norm().item()
                    }
            return hook

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                hook = param.register_hook(hook_fn(name))
                self.hooks.append(hook)

        return grad_stats

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

class PerformanceProfiler:
    def __init__(self, model: torch.nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device

    def profile_model(self, input_shape: tuple, warmup: int = 10, repeats: int = 100):
        self.model.eval()
        dummy_input = torch.randn(input_shape).to(self.device)

        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = self.model(dummy_input)

        # Profile
        times = []
        with torch.no_grad():
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            for _ in range(repeats):
                start_time = time.time()
                _ = self.model(dummy_input)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # Convert to ms

        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'p95_time': np.percentile(times, 95),
            'p99_time': np.percentile(times, 99)
        }

    def profile_memory(self, input_shape: tuple):
        self.model.eval()
        dummy_input = torch.randn(input_shape).to(self.device)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

            with torch.no_grad():
                _ = self.model(dummy_input)

            memory_used = torch.cuda.max_memory_allocated() / 1024**3  # Convert to GB
            return {
                'gpu_memory_gb': memory_used,
                'gpu_memory_mb': memory_used * 1024
            }
        else:
            return {'error': 'CUDA not available'}

    def profile_flops(self, input_shape: tuple):
        try:
            from thop import profile
            dummy_input = torch.randn(input_shape)
            flops, params = profile(self.model, inputs=(dummy_input,))
            return {
                'flops': flops,
                'params': params,
                'flops_gb': flops / 1e9
            }
        except ImportError:
            return {'error': 'thop not installed'}

class ModelComparator:
    def __init__(self, models: Dict[str, torch.nn.Module], device: str = 'cpu'):
        self.models = {name: model.to(device) for name, model in models.items()}
        self.device = device
        self.results = {}

    def compare_models(self, input_shape: tuple, dataset_loader = None):
        profiler = PerformanceProfiler(None, self.device)

        for name, model in self.models.items():
            profiler.model = model
            print(f"Profiling model: {name}")

            # Performance metrics
            self.results[name] = {
                'performance': profiler.profile_model(input_shape),
                'memory': profiler.profile_memory(input_shape),
                'flops': profiler.profile_flops(input_shape)
            }

            # Accuracy metrics if dataset provided
            if dataset_loader:
                self.results[name]['accuracy'] = self.evaluate_model(model, dataset_loader)

        return self.results

    def evaluate_model(self, model: torch.nn.Module, data_loader):
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = model(data)
                _, predicted = outputs.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        accuracy = 100. * correct / total
        return {'accuracy': accuracy}

    def generate_report(self):
        report = "Model Comparison Report\n"
        report += "=" * 50 + "\n\n"

        for model_name, results in self.results.items():
            report += f"Model: {model_name}\n"
            report += "-" * 30 + "\n"

            # Performance
            if 'performance' in results:
                perf = results['performance']
                report += f"  Mean inference time: {perf['mean_time']:.2f} ms\n"
                report += f"  P95 inference time: {perf['p95_time']:.2f} ms\n"

            # Memory
            if 'memory' in results and 'gpu_memory_gb' in results['memory']:
                report += f"  GPU memory: {results['memory']['gpu_memory_gb']:.2f} GB\n"

            # FLOPs
            if 'flops' in results and 'flops_gb' in results['flops']:
                report += f"  FLOPs: {results['flops']['flops_gb']:.2f} GFLOPs\n"

            # Accuracy
            if 'accuracy' in results:
                report += f"  Accuracy: {results['accuracy']['accuracy']:.2f}%\n"

            report += "\n"

        return report

# Usage examples
if __name__ == "__main__":
    # Create models to compare
    models = {
        'resnet18': torch.hub.load('pytorch/vision', 'resnet18', pretrained=True),
        'resnet50': torch.hub.load('pytorch/vision', 'resnet50', pretrained=True),
        'vgg16': torch.hub.load('pytorch/vision', 'vgg16', pretrained=True)
    }

    # Compare models
    comparator = ModelComparator(models, device='cuda')
    results = comparator.compare_models(input_shape=(1, 3, 224, 224))
    report = comparator.generate_report()

    print(report)

    # Save results
    with open('model_comparison_report.txt', 'w') as f:
        f.write(report)
```

This comprehensive implementation guide provides production-ready code for modern deep learning workflows, including advanced training techniques, transfer learning, computer vision applications, deployment strategies, and monitoring tools. Each implementation follows best practices and includes detailed documentation for easy integration into real-world projects.