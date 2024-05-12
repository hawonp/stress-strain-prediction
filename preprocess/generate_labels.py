import polars as pl
from loguru import logger

from settings.config import CONFIGURATION


def generate_labels():
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
    circle_labels = pl.read_excel(
        f"{circle_raw_data_dir}/labels.xlsx",
        read_options={
            "has_header": False,
        },
    )

    # drop the first column
    circle_labels = circle_labels.drop("column_1")

    for series in circle_labels:
        data = series.to_numpy()

    # extract each column and put into a tuple

    logger.info("Generating labels")
