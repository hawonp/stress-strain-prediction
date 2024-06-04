import matplotlib.pyplot as plt
import torch
import torch.utils
import torch.utils.data
from loguru import logger
from torchvision.transforms import v2

from dataloader.custom_dataloader import CustomDataset, LabelLoader
from model.neural_network import NeuralNetwork
from settings.config import CONFIGURATION


def evaluate_random_test_sample():
    # initialize variables
    model = NeuralNetwork(device="cuda")
    optimizer = torch.optim.SGD(model.parameters(), lr=CONFIGURATION.learning_rate)
    loss_function = torch.nn.MSELoss()

    # load checkpoint
    checkpoint = torch.load("./output/take2/best_save.pth")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    running_loss = checkpoint["testing_loss"]

    logger.info("Loading labels...")
    label_loader = LabelLoader()
    _, test_labels = label_loader.get()

    # load dataset
    logger.info("Loading dataset...")
    testing_dataset = CustomDataset(labels=test_labels)

    # load dataloader
    logger.info("Loading dataloader...")

    test_dataloader = torch.utils.data.DataLoader(
        dataset=testing_dataset,
        batch_size=CONFIGURATION.batch_size,
        shuffle=True,
    )

    # read in one batch

    # make prediction
    model.eval()
    data = next(iter(test_dataloader))
    transforms = v2.Compose(
        [
            v2.ToDtype(torch.float32),
        ]
    )
    img, stiffness, strength, hardness = data[0], data[1], data[2], data[3]

    # transform data
    img = transforms(img)
    stiffness = transforms(stiffness)
    strength = transforms(strength)
    hardness = transforms(hardness)

    # turn stiffness, strength, and toughness into a 1x3 tensor
    labels = torch.stack((stiffness, strength, hardness), dim=1)

    prediction = model(img.to("cuda"))
    print(f"Actual: {labels}")
    print(f"Predicted: {prediction}")

    loss = loss_function(prediction, labels.to("cuda"))
    running_loss += loss.item()

    testing_loss = running_loss / len(test_dataloader.dataset)
    print(f"Loss: {testing_loss}")

    # save actual and predicted values into a csv
    for i in range(len(labels)):
        # generate image from tensor
        image_tensor = img[i]
        numpy_array = image_tensor.numpy()

        plt.imsave(f"./output/images/{i}.png", numpy_array[0], cmap="gray")

        with open("./output/actual_predicted.csv", "a") as f:
            f.write(
                f"{i},{labels[i][0].item()},{labels[i][1].item()},{labels[i][2].item()},{prediction[i][0].item()},{prediction[i][1].item()},{prediction[i][2].item()}\n"
            )


if __name__ == "__main__":
    evaluate_random_test_sample()
