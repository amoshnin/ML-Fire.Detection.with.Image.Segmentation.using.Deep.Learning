"""
    File Name : TrainConfig.py
    File Description : TrainConfig class for Configuration for training on the dataset. Derives from the base Config class and overrides some values.
"""

from mrcnn.config import Config

class TrainConfig(Config):
    """Configuration for training on the dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "object"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + Fire (No. of classes)
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100
    # Skip detections with < 50% confidence
    DETECTION_MIN_CONFIDENCE = 0.5

