"""
    File Name : References.py
    File Description : References class for keeping the constants and reference path to dataset and backbone weights
"""

import os
import sys

class References:
    """ Training Mode"""
    # Root directory of the project
    ROOT_DIR = os.path.abspath("./../")
    # Import Mask RCNN
    sys.path.append(ROOT_DIR)  # To find local version of the library
    # Path to trained weights file
    COCO_WEIGHTS_PATH = "/input/mask_rcnn_coco.h5"
    # Directory to save logs and model checkpoints
    DEFAULT_LOGS_DIR = "/output/logs"

    """ Inferencing Mode """
    # Directory to save logs and trained model
    MODEL_DIR = ROOT_DIR + "/output"

    # Path to trained weights
    # You can download this file from the Releases page
    # https://github.com/matterport/Mask_RCNN/releases
    WEIGHTS_PATH = ROOT_DIR + "/output/mask_rcnn_object_0020.h5"

    # Device to load the neural network on.
    # Useful if you're training a model on the same
    # machine, in which case use CPU and leave the
    # GPU for training.
    DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

    # Inspect the model in training or inference modes
    # values: 'inference' or 'training'
    TEST_MODE = "inference"

    CUSTOM_DIR = ROOT_DIR + "/input/dataset"
