from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self, device):
        super(NeuralNetwork, self).__init__()
        self.model = nn.Sequential(  # input size is 1x800x800
            # layer one
            nn.Conv2d(1, 5, 5),  # output size is 5x796x796
            nn.MaxPool2d(2),  # output size is 5x398x398
            nn.ReLU(),
            # layer two
            nn.Conv2d(5, 10, 5),  # output size is 10x394x394
            nn.MaxPool2d(2),  # output size is 10x197x197
            nn.ReLU(),
            # layer three
            nn.Conv2d(10, 20, 5),  # output size is 20x193x193
            nn.MaxPool2d(2),  # output size is 20x96x96
            nn.ReLU(),
            # layer four
            nn.Conv2d(20, 20, 5),  # output size is 20x92x92
            nn.MaxPool2d(2),  # output size is 20x46x46
            nn.ReLU(),
            # layer five
            nn.Conv2d(20, 20, 5),  # output size is 20x42x42
            nn.MaxPool2d(2),  # output size is 20x21x21
            nn.ReLU(),
            # flatten the tensor
            nn.Flatten(),
            nn.Linear(20 * 21 * 21, 3),
        )
        self.model.to(device)

    def forward(self, input):
        return self.model(input)
