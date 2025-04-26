import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob


def get_data():
    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomAffine(degrees=45, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    def create_train(folder_path):
        full_train_data = []
        for i in range(50):
            clas = i
            train_folder_path = 'train_butterflies/train_split/class_' + str(i) + '/*.jpg'
            for image_path in glob.glob(train_folder_path):
                image = Image.open(image_path)
                for _ in range(2):
                    full_train_data.append((transform_train(image), clas))
        return full_train_data

    def create_test(folder_path):
        full_test_data = []
        file_indices = []
        for image_path in glob.glob(folder_path):
            image = Image.open(image_path)
            full_test_data.append(transform_test(image))
            file_index = int(image_path.split('\\')[-1].split('.')[0])
            file_indices.append(file_index)
        return full_test_data, file_indices


    train_folder_path = 'train_butterflies/train_split/'
    test_folder_path = 'test_butterflies/valid/*.jpg'
    train = create_train(train_folder_path)
    test = create_test(test_folder_path)
    return train, test