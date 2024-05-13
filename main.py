import matplotlib.pyplot as plt
import torch
import torch.utils
import torch.utils.data
from loguru import logger

from dataloader.custom_dataloader import CustomDataset, TrainTestSplitter
from model.neural_network import NeuralNetwork
from model.trainer import Trainer
from settings.config import CONFIGURATION


def main():
    # load device
    logger.info("Starting...")
    logger.info("Loading device...")
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info(f"Using: {device}")

    # load configuration
    logger.info("Loading configuration...")
    logger.info(f"Current configuration: {CONFIGURATION.model_dump()}")

    # separate labels into training and testing according to training_split and test_split
    logger.info("Loading labels...")
    splitter = TrainTestSplitter()
    training_labels, test_labels = splitter.split()

    # load dataset
    logger.info("Loading dataset...")
    training_dataset = CustomDataset(labels=training_labels)
    testing_dataset = CustomDataset(labels=test_labels)

    # load dataloader
    logger.info("Loading dataloader...")

    train_dataloader = torch.utils.data.DataLoader(
        dataset=training_dataset,
        batch_size=CONFIGURATION.batch_size,
        shuffle=True,
    )

    test_dataload = torch.utils.data.DataLoader(
        dataset=testing_dataset,
        batch_size=CONFIGURATION.batch_size,
        shuffle=True,
    )
    logger.info(f"Number of training batches: {len(train_dataloader)}")
    logger.info(f"Number of testing batches: {len(test_dataload)}")

    # Display image and label.
    train_features, stiffness, strength, toughness = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    img = train_features[0].squeeze()
    plt.imshow(img, cmap="gray")
    plt.show()

    # load model
    logger.info("Loading model...")
    model = NeuralNetwork(device=device)

    # load optimizer
    logger.info("Loading optimizer...")
    optimizer = torch.optim.SGD(model.parameters(), lr=CONFIGURATION.learning_rate)

    # load loss function
    logger.info("Loading loss function...")
    loss_function = torch.nn.CrossEntropyLoss()

    # load trainer
    logger.info("Loading trainer...")
    trainer = Trainer(
        device=device,
        optimizer=optimizer,
        loss_function=loss_function,
        model=model,
        epochs=CONFIGURATION.epochs,
        train_loader=train_dataloader,
    )

    # train model
    logger.info("Training model...")
    trainer.train()

    # finish
    logger.info("Done!")


if __name__ == "__main__":
    main()
