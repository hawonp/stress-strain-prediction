import polars as pl
import torch
from torchvision.io import read_image  # type: ignore

from settings.config import CONFIGURATION


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, labels):
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = read_image(self.labels[idx, 0])
        stiffness = self.labels[idx, 1]
        strength = self.labels[idx, 2]
        toughness = self.labels[idx, 3]
        return image, stiffness, strength, toughness


class LabelLoader:
    def __init__(self):
        self.train_labels = pl.read_csv(
            f"./{CONFIGURATION.data_dir}/train_labels.csv", has_header=False
        )
        self.test_labels = pl.read_csv(
            f"./{CONFIGURATION.data_dir}/test_labels.csv", has_header=False
        )

    def get(self):
        return self.train_labels, self.test_labels
