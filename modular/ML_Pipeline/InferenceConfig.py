"""
    File Name : InferenceConfig.py
    File Description : InferenceConfig class for loading the inference config for maskrcnn model
"""

from .TrainConfig import TrainConfig

config = TrainConfig()

class InferenceConfig(config.__class__):

    # Running image detection on a single image at a time
    NAME = "object"

    GPU_COUNT = 1

    IMAGES_PER_GPU = 1

    DETECTION_MIN_CONFIDENCE = 0.7