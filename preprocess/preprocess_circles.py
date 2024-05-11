import os

import cv2 as cv
from loguru import logger

from settings.config import CONFIGURATION


def apply_material_properties():
    logger.info("Applying material properties to circles images")
    raw_data_dir = CONFIGURATION.raw_data_dir
    images_path = f"./{raw_data_dir}/circles/images"
    processed_images_path = f"./{raw_data_dir}/circles/processed_images"
    white_circle_value = CONFIGURATION.circle_white
    black_space_value = CONFIGURATION.circle_black

    logger.info(f"White circle value: {white_circle_value}")
    logger.info(f"Black space value: {black_space_value}")

    image_files = [
        f
        for f in os.listdir(images_path)
        if os.path.isfile(os.path.join(images_path, f))
    ]

    for image_path in image_files:
        full_path = os.path.join(images_path, image_path)

        image = cv.imread(full_path, cv.IMREAD_GRAYSCALE)

        # parse only image number
        image_number = image_path.split("-")[2].split("_")[0]
        processed_image_path = f"{processed_images_path}/{image_number}.png"

        # turn 255 to white_circle_value
        # turn 0 to black_space_value
        image[image == 255] = white_circle_value
        image[image == 0] = black_space_value
        cv.imwrite(processed_image_path, image)

    logger.info("Material properties applied to circles images")
