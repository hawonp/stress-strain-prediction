import matplotlib.pyplot as plt
import torch
import torch.utils
import torch.utils.data
from loguru import logger

from dataloader.custom_dataloader import CustomDataset, LabelLoader
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
    logger.debug(f"Current configuration: {CONFIGURATION.model_dump()}")

    # separate labels into training and testing according to training_split and test_split
    logger.info("Loading labels...")
    label_loader = LabelLoader()
    training_labels, test_labels = label_loader.get()

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

    test_dataloader = torch.utils.data.DataLoader(
        dataset=testing_dataset,
        batch_size=CONFIGURATION.batch_size,
        shuffle=True,
    )

    # load model
    logger.info("Loading model...")
    model = NeuralNetwork(device=device)

    # load optimizer
    logger.info("Loading optimizer...")
    optimizer = torch.optim.SGD(model.parameters(), lr=CONFIGURATION.learning_rate)

    # load loss function
    logger.info("Loading loss function...")
    loss_function = torch.nn.MSELoss()

    # load trainer
    logger.info("Loading trainer...")
    trainer = Trainer(
        device=device,
        optimizer=optimizer,
        loss_function=loss_function,
        model=model,
        epochs=CONFIGURATION.epochs,
        train_loader=train_dataloader,
        test_loader=test_dataloader,
    )

    # train model
    logger.info("Training model...")

    losses = []
    for i in range(trainer.epochs):
        logger.info(f"Epoch {i + 1}")
        training_loss = trainer.train()
        testing_loss = trainer.test()
        losses.append((training_loss, testing_loss))

    # finish
    logger.info("Done!")

    # save losses
    logger.info("Saving losses...")
    with open("losses.txt", "w") as f:
        for loss in losses:
            f.write(f"{loss[0]},{loss[1]}\n")

    # plot losses and save
    logger.info("Plotting losses...")
    plt.plot([x[0] for x in losses], label="Training Loss")
    plt.plot([x[1] for x in losses], label="Testing Loss")
    plt.legend()
    plt.savefig("losses.png")


if __name__ == "__main__":
    main()
