# CIFAR-10 Image Classification

This project uses the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) to build a Convolutional Neural Network (CNN) in PyTorch that classifies images into one of ten object categories. I completed this project in August 2025, implementing a full pipeline for image preprocessing, model design, training, evaluation, and prediction.

## Project Overview

The goal is to predict the object class (e.g., airplane, dog, truck) of an image based on its pixel data. The model uses a custom CNN architecture trained on the CIFAR-10 dataset and is able to make predictions on both test images and external images (e.g., a custom image of a dog).

## Dataset

- CIFAR-10 contains 60,000 color images (32x32 pixels) across 10 classes.
- Split:
  - 50,000 images for training
  - 10,000 images for testing
- Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

## Tools and Libraries

- Python
- PyTorch (`torch`, `torchvision`)
- PIL (for loading custom test images)
- matplotlib (optional visualization)

## Process and Methodology

### 1. Data Loading and Augmentation
- Used `torchvision.datasets.CIFAR10` to load the dataset
- Applied data augmentation to the training set:
  - `RandomHorizontalFlip`: randomly flips images left-right
  - `RandomRotation`: rotates images slightly to introduce variation
  - `ToTensor` and `Normalize`: standardizes pixel values
- Applied only normalization to the test set to preserve evaluation consistency

### 2. Model Architecture
- Built a custom CNN using `nn.Module` with:
  - Multiple convolutional layers (ReLU + MaxPool)
  - Fully connected (linear) layers
  - Dropout for regularization

### 3. Training
- Used `CrossEntropyLoss` for multi-class classification
- Optimized with `Adam` (learning rate = 0.001)
- Trained for 30 epochs

### 4. Evaluation
- Measured test set accuracy after each epoch
- Printed predictions on a few custom external images (e.g., `dog.jpg`)

## Final Model Performance

- Highest Accuracy Achieved: ~69%
- Accuracy After Augmentation: ~64% (further tuning needed)

## Files in This Project

- `ImageClassification.ipynb`: main notebook containing all code
- `trained_net.pth`: saved model weights after training
- `dog.jpg`, `airplane.jpg`: custom test images
- `README (as notebook comments)`: summary of the project, purpose, and methods

## Timeline

Project completed in early August 2025.

## Future Improvements

- Tune the CNN architecture (more filters, different activation functions)
- Add batch normalization to stabilize training
- Perform more extensive hyperparameter search (e.g., batch size, dropout rate)
- Use a learning rate scheduler to adjust learning rate during training
- Try pretrained models (e.g., ResNet18 from torchvision)

---
