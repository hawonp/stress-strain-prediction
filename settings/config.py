from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["CONFIGURATION"]


class Configuration(BaseSettings):
    def __init__(self):
        super().__init__()

    # config
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # directories
    raw_data_dir: str = Field(validation_alias="raw_data_dir")
    data_dir: str = Field(validation_alias="data_dir")
    output_dir: str = Field(validation_alias="output_dir")

    # train/validation/test split
    training_split: float = Field(validation_alias="TRAINING_SPLIT")
    test_split: float = Field(validation_alias="TEST_SPLIT")

    # hyperparameters
    batch_size: int = Field(validation_alias="BATCH_SIZE")
    learning_rate: float = Field(validation_alias="LEARNING_RATE")
    epochs: int = Field(validation_alias="EPOCHS")
    save_every: int = Field(validation_alias="SAVE_EVERY")

    # material properties
    circle_white: float = 15.00
    circle_black: float = 3.17
    square_hard: float = 2.1
    square_soft: float = 0.45


CONFIGURATION = Configuration()
