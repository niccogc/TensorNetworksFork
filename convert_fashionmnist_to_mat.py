#!/usr/bin/env python3
"""
Convert FashionMNIST dataset to MATLAB .mat format.
Format specification from: https://github.com/sunsided/mnist-matlab
"""

import numpy as np
import scipy.io as sio
from torchvision import datasets, transforms
import os

def convert_fashionmnist_to_mat(output_path='fashionmnist.mat'):
    """
    Convert FashionMNIST dataset to MATLAB .mat format.

    Args:
        output_path (str): Path to save the .mat file
    """
    # Download FashionMNIST dataset
    print("Downloading FashionMNIST dataset...")

    # Transform to convert PIL Image to tensor and normalize to [0,1]
    transform = transforms.Compose([transforms.ToTensor()])

    # Download training and test sets
    train_dataset = datasets.FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    print("Converting training data...")
    # Convert training data
    train_images = []
    train_labels = []

    for i, (image, label) in enumerate(train_dataset):
        if i % 10000 == 0:
            print(f"Processing training image {i}/{len(train_dataset)}")

        # Convert tensor to numpy and squeeze to remove channel dimension
        img_np = image.numpy().squeeze()
        train_images.append(img_np)
        train_labels.append(label)

    # Stack images into 3D array (height x width x num_images)
    train_images = np.stack(train_images, axis=2)
    train_labels = np.array(train_labels)

    print("Converting test data...")
    # Convert test data
    test_images = []
    test_labels = []

    for i, (image, label) in enumerate(test_dataset):
        if i % 2000 == 0:
            print(f"Processing test image {i}/{len(test_dataset)}")

        # Convert tensor to numpy and squeeze to remove channel dimension
        img_np = image.numpy().squeeze()
        test_images.append(img_np)
        test_labels.append(label)

    # Stack images into 3D array (height x width x num_images)
    test_images = np.stack(test_images, axis=2)
    test_labels = np.array(test_labels)

    # Create MATLAB structure format
    print("Creating MATLAB structure...")

    # Training structure
    training = {
        'count': len(train_dataset),
        'width': 28,
        'height': 28,
        'images': train_images,
        'labels': train_labels
    }

    # Test structure
    test = {
        'count': len(test_dataset),
        'width': 28,
        'height': 28,
        'images': test_images,
        'labels': test_labels
    }

    # Create final structure
    fashionmnist_data = {
        'training': training,
        'test': test
    }

    # Save to .mat file
    print(f"Saving to {output_path}...")
    sio.savemat(output_path, fashionmnist_data)

    print(f"Successfully converted FashionMNIST to {output_path}")
    print(f"Training set: {training['count']} images")
    print(f"Test set: {test['count']} images")
    print(f"Image dimensions: {training['height']}x{training['width']}")

    # Clean up downloaded data directory if desired
    # import shutil
    # shutil.rmtree('./data')

if __name__ == "__main__":
    convert_fashionmnist_to_mat('/work3/aveno/FashionMNIST/fashionmnist.mat')