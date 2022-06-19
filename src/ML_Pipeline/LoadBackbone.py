"""
    File Name : LoadBackbone.py
    File Description : LoadBackbone class for loading the backbone pretrained weights to extract features
"""
from .References import References
from .TrainConfig import TrainConfig
from mrcnn import model as modellib, utils

class LoadBackbone(References):

    def setup(self):
        """ Loading the pretrained COCO weights as backbone """

        # Loading the MaskRCNN training configuration
        config = TrainConfig()

        # Logs Dir
        logs_dir = self.ROOT_DIR + self.DEFAULT_LOGS_DIR

        # Initializing MaskRCNN Model in training mode
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=logs_dir)

        # Coco Weights Dir
        coco_dir = self.ROOT_DIR + self.COCO_WEIGHTS_PATH

        # Loading model weights as backbone
        model.load_weights(coco_dir, by_name=True, exclude=["mrcnn_class_logits",
                                                                "mrcnn_bbox_fc",
                                                                "mrcnn_bbox",
                                                                "mrcnn_mask"])

        return model, config