# Image Forgery Detection using Custom CNN

This project was developed as part of an interview challenge and explores binary classification of authentic vs. spliced images using a custom CNN architecture inspired by Inception and ResNet.

## Project Overview
- Task: Detect spliced (forged) images vs. authentic ones
- Model: Custom CNN built from scratch using PyTorch
- Training: ~1800 images, trained for ~600 epochs
- Outcome: Achieved solid separation in training; results visualized via accuracy and loss graphs

## What’s Included
- Model architecture (CNN)
- Training script
- Accuracy and loss graphs
- Project documentation

## What’s Not Included
- ❌ Training dataset (proprietary/expired access)
- ❌ Final model weights

## Graphs
![Accuracy](graphs/accuracy.png)
![Loss](graphs/loss.png)

## Future Work
- Retrain on public datasets like CASIA, Columbia or real-world spliced datasets
- Add augmentation and early stopping
