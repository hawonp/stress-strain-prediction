import sys

import torch
from loguru import logger
from torcheval.metrics import R2Score
from torchvision.transforms import v2
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        device,
        optimizer,
        loss_function,
        model,
        train_loader,
        test_loader,
    ):
        self.device = device
        self.opt = optimizer
        self.loss_fn = loss_function
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.best_training_loss = sys.maxsize
        self.best_testing_loss = sys.maxsize
        self.metric = R2Score()

    def save(
        self,
        epoch: int,
        training_loss: float,
        testing_loss: float,
    ):
        if training_loss < self.best_training_loss:
            self.best_training_loss = training_loss
        if testing_loss < self.best_testing_loss:
            logger.info("Saving best model...")
            self.best_testing_loss = testing_loss
            best_save = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.opt.state_dict(),
                "training_loss": training_loss,
                "testing_loss": testing_loss,
            }

            torch.save(best_save, "./output/best_save.pth")

    def save_statistics(self): ...

    def train(self):
        self.model.train()
        running_loss = 0
        transforms = v2.Compose(
            [
                v2.ToDtype(torch.float32),
            ]
        )
        for data in tqdm(self.train_loader):
            # unpack data
            img, stiffness, strength, hardness = data[0], data[1], data[2], data[3]

            # transform data
            img = transforms(img)
            stiffness = transforms(stiffness)
            strength = transforms(strength)
            hardness = transforms(hardness)

            # combine stiffness, strength, and hardness into a 64,3 tensor
            labels = torch.stack((stiffness, strength, hardness), dim=1)

            # forward pass
            self.opt.zero_grad()
            result = self.model(img.to(self.device))

            # calculate loss
            loss = self.loss_fn(result, labels.to(self.device))

            # backward pass
            loss.backward()
            self.opt.step()

            running_loss += loss.item() * img.size(0)

            # compute r2 score
            self.metric.update(result, labels.to(self.device))
            score = float(self.metric.compute())

        training_loss = running_loss / len(self.train_loader.dataset)
        logger.info("Training Loss: " + str(training_loss))
        self.metric.reset()
        return training_loss, score

    def test(self):
        self.model.eval()
        running_loss = 0
        transforms = v2.Compose(
            [
                v2.ToDtype(torch.float32),
            ]
        )
        with torch.no_grad():
            for data in tqdm(self.test_loader):
                # unpack data
                img, stiffness, strength, hardness = data[0], data[1], data[2], data[3]

                # transform data
                img = transforms(img)
                stiffness = transforms(stiffness)
                strength = transforms(strength)
                hardness = transforms(hardness)

                # combine stiffness, strength, and hardness into a 64,3 tensor
                labels = torch.stack((stiffness, strength, hardness), dim=1)

                # forward pass
                result = self.model(img.to(self.device))

                # calculate loss
                loss = self.loss_fn(result, labels.to(self.device))

                running_loss += loss.item()

                # compute r2 score
                self.metric.update(result, labels.to(self.device))
                score = float(self.metric.compute())

        testing_loss = running_loss / len(self.test_loader.dataset)
        logger.info("Testing loss: " + str(testing_loss))
        return testing_loss, score
