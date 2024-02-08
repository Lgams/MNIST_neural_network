MNIST handwritten digit classifier
This repository contains an end-to-end solution for training and evaluating a neural network on the MNIST dataset. The MNIST dataset comprises 28x28 pixel grayscale images of handwritten digits. The project is divided into two main parts:

Training: The train_model.py script builds a simple, yet effective, neural network using PyTorch, trains it on the MNIST training dataset, and saves the model weights upon completion. The model is a fully connected neural network with three layers, using ReLU activation functions and CrossEntropyLoss for classification.

Evaluation: The evaluate_model.py script loads the trained model and evaluates its performance on the MNIST test dataset. It includes functionality to display a grid of handwritten digit images alongside the model's predictions and the true labels for visual verification.

The repository also includes example output images that demonstrate the model's prediction accuracy. These examples serve as a visual testament to the model's capabilities.
