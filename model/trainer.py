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
        self.epo = epochs
        self.train_loader = train_loader

    def train(self):
        for epoch in range(self.epochs):
            cum_loss = 0
            for data in tqdm(self.train_loader):
                self.opt.zero_grad()
                img, gt = data[0], data[1]
                result = self.model(img.to(self.device))
                # result = result.argmax(dim=1)
                loss = self.loss_fn(result, gt.to(self.device))
                loss.backward()
                self.opt.step()
                cum_loss += loss.item()
            print("Cumulative loss: " + str(cum_loss))
