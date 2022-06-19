"""
    File Name : InferenceModel.py
    File Description : InferenceModel class for loading the trained model and making inferences from it
"""
import tensorflow as tf
import mrcnn.model as modellib

from .References import References


class InferenceModel(References):

    def load_model(self, config):
        # LOAD MODEL

        # Setup model in inference mode
        with tf.device(self.DEVICE):
            model = modellib.MaskRCNN(mode="inference", model_dir=self.MODEL_DIR,
                                      config=config)

        # Load COCO weights, Or load the last model you trained
        # Load weights
        print("Loading weights ")
        model.load_weights(self.WEIGHTS_PATH, by_name=True, exclude=["mrcnn_class_logits",
                                                                "mrcnn_bbox_fc",
                                                                "mrcnn_bbox",
                                                                "mrcnn_mask"])
        return model