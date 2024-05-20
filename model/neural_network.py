from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self, device):
        super(NeuralNetwork, self).__init__()
        self.model = nn.Sequential(  # input size is 1x800x800
            # #########################
            # # CONV + MAXPOOL + ReLU #
            # #########################
            # nn.Conv2d(1, 10, 5),  # output size is 10x796x796
            # nn.MaxPool2d(2),  # output size is 10x398x398
            # nn.ReLU(),
            # # #########################
            # # # CONV + MAXPOOL + ReLU #
            # # #########################
            # nn.Conv2d(10, 10, 5),  # output size is 10x394x394
            # nn.MaxPool2d(2),  # output size is 10x197x197
            # nn.ReLU(),
            # # #########################
            # # # CONV + MAXPOOL + ReLU #
            # # #########################
            # nn.Conv2d(10, 10, 5),  # output size is 10x193x193
            # nn.MaxPool2d(2),  # output size is 10x96x96
            # nn.ReLU(),
            # # #########################
            # # # CONV + MAXPOOL + ReLU #
            # # #########################
            # nn.Conv2d(10, 10, 5),  # output size is 10x92x92
            # nn.MaxPool2d(2),  # output size is 10x46x46
            # nn.ReLU(),
            # #####################
            # # REGRESSION OUTPUT #
            # #####################
            # # simplify to 1x3
            # nn.Flatten(),
            # nn.Linear(10 * 46 * 46, 3),
            nn.Conv2d(1, 5, 5),  # output size is 5x796x796
            nn.MaxPool2d(2),  # output size is 5x398x398
            nn.ReLU(),
            nn.Conv2d(5, 10, 5),  # output size is 10x394x394
            nn.MaxPool2d(2),  # output size is 10x197x197
            nn.ReLU(),
            nn.Conv2d(10, 20, 5),  # output size is 20x193x193
            nn.MaxPool2d(2),  # output size is 20x96x96
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(20 * 96 * 96, 3),
        )
        self.model.to(device)

    def forward(self, input):
        return self.model(input)
