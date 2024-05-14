import torch
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
    ):
        self.device = device
        self.opt = optimizer
        self.loss_fn = loss_function
        self.model = model
        self.epochs = epochs
        self.train_loader = train_loader

    def train(self):
        for epoch in range(self.epochs):
            self.model = self.model.train(True)
            print("Epoch: " + str(epoch))
            cum_loss = 0
            for data in tqdm(self.train_loader):
                img, stiffness, hardness, strength = data[0], data[1], data[2], data[3]
                transforms = v2.Compose(
                    [
                        v2.ToDtype(torch.float32),
                    ]
                )
                img = transforms(img)
                stiffness = transforms(stiffness)
                result = self.model(img.to(self.device))
                loss = self.loss_fn(result, stiffness.to(self.device))
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                cum_loss += loss.item()
            print("Cumulative loss: " + str(cum_loss))
