# Quantum Convolutional Neural Network for Brain Tumor Detection

This repository contains an implementation of a Quantum Convolutional Neural Network (QCNN) for detecting brain tumors in MRI images.

## Overview

The model uses a hybrid quantum-classical approach:
1. **Quantum processing**: Brain MRI images are processed using a quantum circuit to extract meaningful features
2. **Classical neural network**: A standard neural network processes the quantum features for final classification

## Dataset

The model uses the Brain Tumor MRI Dataset which contains:
- MRI images of brains with tumors (in the "yes" directory)
- MRI images of healthy brains (in the "no" directory)

## Requirements

- Python 3.6+
- PennyLane (for quantum computing)
- TensorFlow (for classical neural network)
- Additional libraries in `requirements.txt`

## Installation

1. Clone this repository
2. Install dependencies:
```
pip install -r requirements.txt
```

## Usage

Run the main script:
```
python brain_tumor_qcnn.py
```

This will:
1. Load and preprocess the brain MRI dataset
2. Apply quantum convolution to the images
3. Train a hybrid quantum-classical model
4. Evaluate the model and display results

## Implementation Details

- **Quantum Circuit**: Uses PennyLane's quantum simulator with 4 qubits and random rotation layers
- **Quantum Convolution**: Processes 2×2 patches of the downsampled images
- **Classical Model**: A fully connected neural network with dropout layers for classification
- **Visualization**: Includes plots of original images, quantum features, training metrics, and confusion matrix

## Reference

This implementation is based on the quantum convolution approach described in PennyLane's quanvolutional neural networks tutorial, adapted for brain tumor detection in MRI images. 