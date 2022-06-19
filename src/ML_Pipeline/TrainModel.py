"""
    File Name : TrainModel.py
    File Description : TrainModel class for fitting the model with the data and annotations for training set and validation se
"""
class TrainModel:

    def train(self, model, config, dataset_train, dataset_val):
        """Train the model."""

        # Since we're using a small dataset, and starting from
        # COCO trained weights, we don't need to train too long.
        # Also, no need to train all layers, just the heads should do it.
        # Other option can be "all" to train all layers

        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=20,
                    layers='heads')