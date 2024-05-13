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


class TrainTestSplitter:
    def __init__(self):
        self.labels = pl.read_csv(
            f"./{CONFIGURATION.data_dir}/labels.csv", has_header=False
        )
        self.training_split = CONFIGURATION.training_split

    def split(self):
        # split using pure python and not polars
        labels = self.labels.to_numpy()

        # shuffle labels
        torch.manual_seed(0)
        torch.randperm(len(labels))
        labels = labels[torch.randperm(len(labels))]

        split_index = int(len(labels) * self.training_split)
        training_labels = labels[:split_index]
        test_labels = labels[split_index:]

        return training_labels, test_labels
