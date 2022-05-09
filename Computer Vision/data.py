
from __future__ import print_function
from __future__ import division

import torch
import torchvision
from torchvision import transforms
import logging

from torchvision import datasets, models, transforms
import os



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# logger.info("PyTorch Version: ",torch.__version__)
# logger.info("Torchvision Version: ",torchvision.__version__)


def get_data_loaders(data_dir, input_size, batch_size):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    logger.info("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'test']}
    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'test']}

    return image_datasets, dataloaders_dict
