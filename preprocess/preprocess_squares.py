import cv2
import numpy as np
from loguru import logger

from settings.config import CONFIGURATION


def generate_square_images():
    logger.info("Generating square images")
    raw_data_dir = CONFIGURATION.raw_data_dir
    filepath = f"./{raw_data_dir}/squares/input.txt"
    processed_images_path = f"./{raw_data_dir}/squares/processed_images"

    logger.info(f"Hard value: {CONFIGURATION.square_hard}")
    logger.info(f"Soft value: {CONFIGURATION.square_soft}")

    logger.info(f"Reading raw data from {filepath}")

    with open(filepath, "r") as f:
        raw_data = f.readlines()

    # convert raw data to 2d array
    logger.info("Converting raw data to 2d array")
    i = 1
    for line in raw_data:
        logger.info(f"Processing line {i}")
        line = line.strip()
        line = line.split(",")
        matrix = [int(i) for i in line]
        matrix = np.array(matrix).reshape(11, 11)

        # Repeat the matrix to get an 792x792 matrix
        repeat_factor = 800 // 11
        large_matrix = np.repeat(
            np.repeat(matrix, repeat_factor, axis=0), repeat_factor, axis=1
        )

        # Convert to uint8
        large_matrix = (large_matrix).astype(np.uint8)

        large_matrix[large_matrix == 1] = CONFIGURATION.square_hard
        large_matrix[large_matrix == 0] = CONFIGURATION.square_soft

        # scale up image to 800x800
        large_matrix = cv2.resize(
            large_matrix, (800, 800), interpolation=cv2.INTER_NEAREST
        )

        # write image to file
        cv2.imwrite(f"{processed_images_path}/{i}.png", large_matrix)
        i += 1

    logger.info("Square images generated")
