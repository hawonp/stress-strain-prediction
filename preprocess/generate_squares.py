from loguru import logger

from settings.config import CONFIGURATION


def generate_square_images():
    logger.info("Generating square images")
    raw_data_dir = CONFIGURATION.raw_data_dir
    filepath = f"./{raw_data_dir}/squares/input.txt"

    # print(filepath)
    # read in the file
    matrices = []
    with open(filepath, "r") as f:
        raw_data = f.readlines()

    # convert raw data to 2d array
    for line in raw_data:
        line = line.strip()
        line = line.split(",")
        matrix = [int(i) for i in line]
        matrices.append(matrix)

    logger.info("Square images generated")
