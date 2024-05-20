import torch
from loguru import logger
from torchvision.transforms import v2
from tqdm import tqdm  # type: ignore


class Trainer:
    def __init__(
        self,
        device,
        optimizer,
        loss_function,
        model,
        epochs,
        train_loader,
        test_dataload,
    ):
        self.device = device
        self.opt = optimizer
        self.loss_fn = loss_function
        self.model = model
        self.epochs = epochs
        self.train_loader = train_loader
        self.test_dataload = test_dataload

    def train(self):
        self.model.train()
        cum_loss = 0
        for data in tqdm(self.train_loader):
            # unpack data
            img, stiffness, strength, hardness = data[0], data[1], data[2], data[3]
            transforms = v2.Compose(
                [
                    v2.ToDtype(torch.float32),
                ]
            )
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

            # print loss
            # print(loss.data, loss.grad)
            cum_loss += loss.item() * img.size(0)
            # print("Loss: " + str(loss.item()))
        logger.info("Training Loss: " + str(cum_loss / len(self.train_loader)))

    def test(self):
        self.model.eval()
        test_loss = 0

        with torch.no_grad():
            for data in tqdm(self.test_dataload):
                # unpack data
                img, stiffness, strength, hardness = data[0], data[1], data[2], data[3]
                transforms = v2.Compose(
                    [
                        v2.ToDtype(torch.float32),
                    ]
                )
                # transform data
                img = transforms(img)
                stiffness = transforms(stiffness)
                strength = transforms(strength)
                hardness = transforms(hardness)

                # combine stiffness, strength, and hardness into a 64,3 tensor
                labels = torch.stack((stiffness, strength, hardness), dim=1)

                # forward pass
                result = self.model(img.to(self.device))

                # calculate lossw
                loss = self.loss_fn(result, labels.to(self.device))

                # calculate accuracy
                test_loss += loss.item()

        test_loss /= len(self.train_loader)
        logger.info("Testing loss: " + str(test_loss))
