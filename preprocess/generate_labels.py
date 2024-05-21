import polars as pl
import torch
from loguru import logger

from settings.config import CONFIGURATION


def generate_labels():
    logger.info("Generating labels")
    data_dir = f"./{CONFIGURATION.data_dir}"
    square_raw_data_dir = f"./{CONFIGURATION.raw_data_dir}/squares"
    circle_raw_data_dir = f"./{CONFIGURATION.raw_data_dir}/circles"

    square_processed_images_dir = f"{square_raw_data_dir}/processed_images"
    circle_processed_images_dir = f"{circle_raw_data_dir}/processed_images"

    logger.info(f"square_processed_images_dir {square_processed_images_dir}")
    logger.info(f"circle_processed_images_dir {circle_processed_images_dir}")

    # read in square stiffness.txt
    with open(f"{square_raw_data_dir}/stiffness.txt", "r") as f:
        square_stiffness = f.readlines()
    square_stiffness = [float(str(x).strip()) for x in square_stiffness]
    # read in square strength.txt
    with open(f"{square_raw_data_dir}/strength.txt", "r") as f:
        square_strength = f.readlines()
    square_strength = [float(str(x).strip()) for x in square_strength]

    # read in square toughness.txt
    with open(f"{square_raw_data_dir}/toughness.txt", "r") as f:
        square_toughness = f.readlines()
    square_toughness = [float(str(x).strip()) for x in square_toughness]

    # generate a list of tuples that contains the image path, stiffness, strength, and toughness
    square_labels = []
    for i in range(100000):
        square_labels.append(
            (
                f"{square_processed_images_dir}/{i+1}.png",
                square_stiffness[i],
                square_strength[i],
                square_toughness[i],
            )
        )

    # read in circle labels
    circle_dataframe = pl.read_excel(
        f"{circle_raw_data_dir}/labels.xlsx",
        read_options={
            "has_header": False,
        },
    )

    # drop the first column
    circle_dataframe = circle_dataframe.drop("column_1")

    # extract each column and put into a tuple
    i = 1
    circle_labels = []
    base_start = 100000
    for series in circle_dataframe:
        data = series.to_numpy()
        circle_labels.append(
            (
                f"{circle_processed_images_dir}/{base_start+i}.png",
                data[0],
                data[1],
                data[2],
            )
        )

        i += 1

    # shuffle the square labels
    torch.manual_seed(0)
    torch.randperm(len(square_labels))
    square_labels = square_labels[torch.randperm(len(square_labels))]

    # split square labels into training and testing
    num_of_square_labels = len(square_labels)
    num_of_training_square_labels = int(
        num_of_square_labels * CONFIGURATION.training_split
    )

    train_square_labels = square_labels[:num_of_training_square_labels]
    test_square_labels = square_labels[num_of_training_square_labels:]

    # shuffle the circle labels
    torch.manual_seed(0)
    torch.randperm(len(circle_labels))
    circle_labels = circle_labels[torch.randperm(len(circle_labels))]

    # split circle labels into training and testing
    num_of_circle_labels = len(circle_labels)
    num_of_training_circle_labels = int(
        num_of_circle_labels * CONFIGURATION.training_split
    )

    train_circle_labels = circle_labels[:num_of_training_circle_labels]
    test_circle_labels = circle_labels[num_of_training_circle_labels:]

    # combine labels
    train_labels = train_square_labels + train_circle_labels
    test_labels = test_square_labels + test_circle_labels

    # write as csv to disk in the data directory
    pl.DataFrame(train_labels).write_csv(
        f"{data_dir}/train_labels.csv", include_header=False
    )
    pl.DataFrame(test_labels).write_csv(
        f"{data_dir}/test_labels.csv", include_header=False
    )
