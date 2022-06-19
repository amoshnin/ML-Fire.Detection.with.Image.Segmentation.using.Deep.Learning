"""
    File Name : Engine.py
    File Description : Main class for starting different parts and processes of training and inference lifecycle
"""


from ML_Pipeline.TrainModel import TrainModel
from ML_Pipeline.LoadDataset import LoadDataset
from ML_Pipeline.LoadBackbone import LoadBackbone
from ML_Pipeline.References import References
from ML_Pipeline.InferenceConfig import InferenceConfig
from ML_Pipeline.InferenceModel import InferenceModel
from ML_Pipeline.VisualizeMask import VisualizeMask

""" Training Phase """

def train():
    global model, config
    # Loading Training dataset.
    dataset_train = LoadDataset()
    dataset_train.load_custom(References.ROOT_DIR + "/input/dataset", "train")
    dataset_train.prepare()
    # Loading Validation dataset
    dataset_val = LoadDataset()
    dataset_val.load_custom(References.ROOT_DIR + "/input/dataset", "val")
    dataset_val.prepare()
    # Setting up the backbone to train
    backbone = LoadBackbone()
    model, config = backbone.setup()
    # Training the model
    model_train = TrainModel()
    model_train.train(model, config, dataset_train, dataset_val)


""" Inference Phase"""

def inference_model():
    global config, model
    ## Testing

    # Setting up the configuration for inference model
    config = InferenceConfig()
    config.display()
    # Load validation dataset
    dataset = LoadDataset()
    dataset.load_custom(References.CUSTOM_DIR, "val")
    # Must call before using the dataset
    dataset.prepare()
    # Loading weights from the trained model
    inference = InferenceModel()
    model = inference.load_model(config)
    # Predicting the mask and plotting it over test images
    predict = VisualizeMask()
    predict.predict_and_display(dataset, config, model)



# Call to start training
train()

# Call to infer prediction
# inference_model()







