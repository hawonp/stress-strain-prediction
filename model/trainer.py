import torch
from loguru import logger
from torchvision.transforms import v2
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        device,
        optimizer,
        loss_function,
        model,
        epochs,
        train_loader,
        test_loader,
    ):
        self.device = device
        self.opt = optimizer
        self.loss_fn = loss_function
        self.model = model
        self.epochs = epochs
        self.train_loader = train_loader
        self.test_loader = test_loader

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
            # print(running_loss)

        training_loss = running_loss / len(self.train_loader.dataset)
        logger.info("Training Loss: " + str(training_loss))
        return training_loss

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
                # print(running_loss)

        testing_loss = running_loss / len(self.test_loader.dataset)
        logger.info("Testing loss: " + str(testing_loss))
        return testing_loss
