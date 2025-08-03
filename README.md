**# -----------------------------------------------
# CIFAR-10 Image Classification using CNN
# -----------------------------------------------
# This project uses a Convolutional Neural Network (CNN) 
# to classify images from the CIFAR-10 dataset using PyTorch.

# Features:
# - Loads and preprocesses CIFAR-10 images
# - Builds a CNN using nn.Module
# - Applies image augmentation to improve generalization
# - Trains the model using the Adam optimizer and CrossEntropyLoss
# - Evaluates the model's accuracy on the test set
# - Predicts the class of new custom images (e.g., dog.jpg, airplane.jpg)

# Files in this project:
# - ImageClassification.ipynb: Main Jupyter Notebook with all code
# - trained_net.pth: Saved model weights after training
# - dog.jpg / airplane.jpg: Example images for testing model predictions

# Dataset Info:
# - CIFAR-10 contains 60,000 color images of size 32x32
# - 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
# - Split: 50,000 images for training, 10,000 for testing

# Evaluation:
# - The model prints overall accuracy after testing on unseen data
# - Prints class name predictions for custom images

# Tools Used:
# - PyTorch (torch, torchvision)
# - PIL for image loading
# - matplotlib for optional data visualization
**
