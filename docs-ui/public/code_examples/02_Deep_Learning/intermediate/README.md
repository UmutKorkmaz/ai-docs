# Deep Learning - Intermediate Code Examples

## CNN Image Classification with PyTorch

**File**: `cnn_image_classification_pytorch.py`

### Overview
Complete implementation of a Convolutional Neural Network (CNN) for image classification using the CIFAR-10 dataset. This example demonstrates:
- Building a CNN architecture from scratch
- Data augmentation techniques
- Training loop with validation
- Model evaluation and visualization
- Saving best models

### Requirements
```bash
pip install -r requirements.txt
```

### Usage
```bash
python cnn_image_classification_pytorch.py
```

### Expected Output
- Training progress with loss and accuracy metrics
- Best model saved to `./models/best_model.pth`
- Visualization of predictions: `predictions.png`
- Training curves: `training_curves.png`

### Architecture
```
SimpleCNN:
  3 Convolutional blocks (Conv2d -> ReLU -> MaxPool)
  3 Fully connected layers
  Dropout regularization
  ~500K parameters
```

### Expected Performance
- Training time: ~10-15 minutes on GPU
- Test accuracy: ~70-75% after 20 epochs
- Can be improved with deeper architectures or more epochs

### Learning Points
1. **Data Augmentation**: Random crop and horizontal flip improve generalization
2. **Dropout**: Prevents overfitting in fully connected layers
3. **Batch Normalization**: Can add for faster convergence (exercise)
4. **Learning Rate Scheduling**: Can add for better optimization (exercise)

### Exercises
1. Add batch normalization after each convolutional layer
2. Implement learning rate scheduling
3. Try different architectures (ResNet-inspired blocks)
4. Experiment with different optimizers (SGD with momentum)
5. Increase model capacity and train for more epochs

### Related Resources
- [Section 02: Deep Learning](../../../02_Advanced_Deep_Learning/00_Overview.md)
- [Section 04: Computer Vision](../../../04_Computer_Vision/00_Overview.md)
- [Interactive Notebook: CNN Training](../../../interactive/notebooks/02_Advanced_Deep_Learning/)

### References
- PyTorch Documentation: https://pytorch.org/docs/
- CIFAR-10 Dataset: https://www.cs.toronto.edu/~kriz/cifar.html
