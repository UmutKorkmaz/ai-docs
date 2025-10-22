"""
Title: CNN Image Classification with PyTorch
Section: 02. Advanced Deep Learning
Difficulty: Intermediate
Description: Build and train a Convolutional Neural Network for image classification using PyTorch

Prerequisites:
- Python 3.8+
- PyTorch, torchvision
- Basic understanding of neural networks
- Familiarity with image data

Learning Objectives:
- Build a CNN architecture from scratch
- Train a model on CIFAR-10 dataset
- Implement data augmentation
- Evaluate model performance
- Visualize results

Usage:
    python cnn_image_classification_pytorch.py

Author: AI Documentation Team
Date: October 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configuration
CONFIG = {
    'seed': 42,
    'batch_size': 128,
    'num_epochs': 20,
    'learning_rate': 0.001,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_classes': 10,
    'save_dir': './models'
}

# Set random seeds for reproducibility
torch.manual_seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])


class SimpleCNN(nn.Module):
    """
    Simple Convolutional Neural Network for image classification.

    Architecture:
    - 2 Convolutional blocks (Conv -> ReLU -> MaxPool)
    - 3 Fully connected layers
    - Dropout for regularization
    """

    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

        # Activation
        self.relu = nn.ReLU()

    def forward(self, x):
        # Convolutional blocks
        x = self.pool(self.relu(self.conv1(x)))  # 32x32 -> 16x16
        x = self.pool(self.relu(self.conv2(x)))  # 16x16 -> 8x8
        x = self.pool(self.relu(self.conv3(x)))  # 8x8 -> 4x4

        # Flatten
        x = x.view(-1, 128 * 4 * 4)

        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


def get_data_loaders():
    """
    Prepare CIFAR-10 data loaders with data augmentation.

    Returns:
        tuple: (train_loader, test_loader, classes)
    """
    # Data augmentation for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Standard transformation for testing
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    trainloader = DataLoader(
        trainset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2
    )

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    testloader = DataLoader(
        testset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2
    )

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, classes


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train model for one epoch.

    Args:
        model: PyTorch model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on

    Returns:
        float: Average loss for the epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Print statistics every 100 mini-batches
        if i % 100 == 99:
            print(f'  [{i + 1}] loss: {running_loss / 100:.3f}, '
                  f'accuracy: {100 * correct / total:.2f}%')
            running_loss = 0.0

    return running_loss / len(train_loader)


def evaluate(model, test_loader, device):
    """
    Evaluate model on test set.

    Args:
        model: PyTorch model
        test_loader: Test data loader
        device: Device to evaluate on

    Returns:
        float: Test accuracy
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def visualize_predictions(model, test_loader, classes, device, num_images=10):
    """
    Visualize model predictions on test images.

    Args:
        model: Trained model
        test_loader: Test data loader
        classes: Class names
        device: Device
        num_images: Number of images to display
    """
    model.eval()

    # Get one batch
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)

    # Get predictions
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

    # Plot images
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()

    for idx in range(num_images):
        img = images[idx].cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = img * 0.5 + 0.5  # Denormalize
        img = np.clip(img, 0, 1)

        axes[idx].imshow(img)
        axes[idx].axis('off')

        true_label = classes[labels[idx]]
        pred_label = classes[predicted[idx]]
        color = 'green' if labels[idx] == predicted[idx] else 'red'

        axes[idx].set_title(f'True: {true_label}\nPred: {pred_label}',
                           color=color, fontsize=10)

    plt.tight_layout()
    plt.savefig('predictions.png', dpi=150, bbox_inches='tight')
    print("\nPredictions saved to 'predictions.png'")
    plt.close()


def main():
    """
    Main training and evaluation pipeline.
    """
    print("=" * 70)
    print("CNN Image Classification with PyTorch")
    print("=" * 70)
    print(f"\nDevice: {CONFIG['device']}")
    print(f"Batch Size: {CONFIG['batch_size']}")
    print(f"Epochs: {CONFIG['num_epochs']}")
    print(f"Learning Rate: {CONFIG['learning_rate']}\n")

    # Create save directory
    Path(CONFIG['save_dir']).mkdir(exist_ok=True)

    # Get data loaders
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader, classes = get_data_loaders()
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}\n")

    # Initialize model
    model = SimpleCNN(num_classes=CONFIG['num_classes']).to(CONFIG['device'])
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

    # Training loop
    print("Starting training...")
    print("-" * 70)

    best_accuracy = 0.0
    train_losses = []
    test_accuracies = []

    for epoch in range(CONFIG['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{CONFIG['num_epochs']}")

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer,
                                CONFIG['device'])
        train_losses.append(train_loss)

        # Evaluate
        test_acc = evaluate(model, test_loader, CONFIG['device'])
        test_accuracies.append(test_acc)

        print(f"  Test Accuracy: {test_acc:.2f}%")

        # Save best model
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(),
                      f"{CONFIG['save_dir']}/best_model.pth")
            print(f"  ✓ New best model saved! (Accuracy: {best_accuracy:.2f}%)")

    print("\n" + "=" * 70)
    print(f"Training completed!")
    print(f"Best Test Accuracy: {best_accuracy:.2f}%")
    print("=" * 70)

    # Visualize predictions
    print("\nGenerating predictions visualization...")
    visualize_predictions(model, test_loader, classes, CONFIG['device'])

    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)

    ax2.plot(test_accuracies)
    ax2.set_title('Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    print("Training curves saved to 'training_curves.png'")

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
