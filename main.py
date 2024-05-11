import torch
from loguru import logger

from settings.config import CONFIGURATION


def load_device() -> str:
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    return device


def main():
    # load device
    logger.info("Starting...")
    logger.info("Loading device...")
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info(f"Using: {device}")

    # load configuration
    logger.info("Loading configuration...")
    logger.info(f"Current configuration: {CONFIGURATION.model_dump()}")

    # finish
    logger.info("Done!")


if __name__ == "__main__":
    main()
