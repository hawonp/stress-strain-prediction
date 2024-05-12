import os

import cv2 as cv
from loguru import logger

from settings.config import CONFIGURATION


def apply_material_properties():
    logger.info("Applying material properties to circles images")
    raw_data_dir = CONFIGURATION.raw_data_dir
    images_path = f"./{raw_data_dir}/circles/images"
    processed_images_path = f"./{raw_data_dir}/circles/processed_images"

    logger.info(f"White circle value: {CONFIGURATION.circle_white}")
    logger.info(f"Black space value: {CONFIGURATION.circle_black}")

    image_files = [
        f
        for f in os.listdir(images_path)
        if os.path.isfile(os.path.join(images_path, f))
    ]
    base_number = 100000
    for image_path in image_files:
        full_path = os.path.join(images_path, image_path)

        image = cv.imread(full_path, cv.IMREAD_GRAYSCALE)

        # parse only image number
        image_number = image_path.split("-")[2].split("_")[0]
        image_number = int(image_number) + base_number
        processed_image_path = f"{processed_images_path}/{image_number}.png"

        # turn 255 to white_circle_value
        # turn 0 to black_space_value
        image[image == 255] = CONFIGURATION.circle_white
        image[image == 0] = CONFIGURATION.circle_black
        cv.imwrite(processed_image_path, image)

    logger.info("Material properties applied to circles images")
