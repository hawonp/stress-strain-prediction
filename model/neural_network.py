from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self, device):
        super(NeuralNetwork, self).__init__()
        # self.model = nn.Sequential(  # input size is 1x800x800
        #     # layer one
        #     nn.Conv2d(1, 5, 5),  # output size is 5x796x796
        #     nn.MaxPool2d(2),  # output size is 5x398x398
        #     nn.ReLU(),
        #     # layer two
        #     nn.Conv2d(5, 10, 5),  # output size is 10x394x394
        #     nn.MaxPool2d(2),  # output size is 10x197x197
        #     nn.ReLU(),
        #     # layer three
        #     nn.Conv2d(10, 20, 5),  # output size is 20x193x193
        #     nn.MaxPool2d(2),  # output size is 20x96x96
        #     nn.ReLU(),
        #     # layer four
        #     nn.Conv2d(20, 20, 5),  # output size is 20x92x92
        #     nn.MaxPool2d(2),  # output size is 20x46x46
        #     nn.ReLU(),
        #     # layer five
        #     nn.Conv2d(20, 20, 5),  # output size is 20x42x42
        #     nn.MaxPool2d(2),  # output size is 20x21x21
        #     nn.ReLU(),
        #     # layer six
        #     nn.Conv2d(20, 20, 5),  # output size is 20x17x17
        #     nn.MaxPool2d(2),  # output size is 20x8x8
        #     nn.ReLU(),
        #     # flatten the tensor
        #     nn.Flatten(),
        #     nn.Linear(20 * 8 * 8, 3),
        # )
        self.model = nn.Sequential(  # input size is 1x800x800
            # layer one
            nn.Conv2d(1, 5, 3),  # output size is 5x798x798
            nn.MaxPool2d(2),  # output size is 5x399x399
            nn.ReLU(),
            # layer two
            nn.Conv2d(5, 10, 3),  # output size is 10x397x397
            nn.MaxPool2d(2),  # output size is 10x198x198
            nn.ReLU(),
            # layer three
            nn.Conv2d(10, 20, 3),  # output size is 20x196x196
            nn.MaxPool2d(2),  # output size is 20x98x98
            nn.ReLU(),
            # layer four
            nn.Conv2d(20, 20, 3),  # output size is 20x96x96
            nn.MaxPool2d(2),  # output size is 20x48x48
            nn.ReLU(),
            # layer five
            nn.Conv2d(20, 20, 3),  # output size is 20x46x46
            nn.MaxPool2d(2),  # output size is 20x23x23
            nn.ReLU(),
            # layer six
            nn.Conv2d(20, 20, 3),  # output size is 20x21x21
            nn.MaxPool2d(2),  # output size is 20x10x10
            nn.ReLU(),
            # layer seven
            nn.Conv2d(20, 20, 3),  # output size is 20x8x8
            nn.MaxPool2d(2),  # output size is 20x4x4
            nn.ReLU(),
            # flatten the tensor
            nn.Flatten(),
            nn.Linear(20 * 4 * 4, 3),
        )
        self.model.to(device)

    def forward(self, input):
        return self.model(input)
