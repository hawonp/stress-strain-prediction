from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self, device):
        super(NeuralNetwork, self).__init__()
        self.model = nn.Sequential(  # input size is 1x800x800
            #########################
            # CONV + MAXPOOL + ReLU #
            #########################
            nn.Conv2d(1, 10, 5),  # output size is 10x796x796
            nn.MaxPool2d(2),  # output size is 10x398x398
            nn.ReLU(),
            # #########################
            # # CONV + MAXPOOL + ReLU #
            # #########################
            # nn.Conv2d(1, 1, 5),  # output size is 1x394x394
            # nn.MaxPool2d(2),  # output size is 1x197x197
            # nn.ReLU(),
            # #########################
            # # CONV + MAXPOOL + ReLU #
            # #########################
            # nn.Conv2d(1, 1, 5),  # output size is 1x193x193
            # nn.MaxPool2d(2),  # output size is 1x96x96
            # nn.ReLU(),
            # #########################
            # # CONV + MAXPOOL + ReLU #
            # #########################
            # nn.Conv2d(1, 1, 5),  # output size is 1x92x92
            # nn.MaxPool2d(2),  # output size is 1x46x46
            # nn.ReLU(),
            #####################
            # REGRESSION OUTPUT #
            #####################
            # simplify to 1x3
            nn.Flatten(),
            nn.Linear(10 * 398 * 398, 3),
        )
        self.model.to(device)

    def forward(self, input):
        return self.model(input)
