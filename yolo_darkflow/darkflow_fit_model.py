# Import packages
import matplotlib.pyplot as plt
import numpy as np
from darkflow.net.build import TFNet
import cv2

"""
    Skript for train model tiny-yolo-voc-2c.cfg,
    and weights tiny-yolo-voc.weights.
    This model was trained on a voc-dataset.
    Config file was modify to 2 classes.
    If you want to change number of training classes you can change it in
    tiny-yolo-voc-2c.cfg file.
    TODO Modify config file to change number classes
    and add formuls to change number of parametrs.
    TODO more installation options.
    If your have a GPU and installed GPU-tensorflow
     use a "GPU" : 1, in option config.

"""
options = {"model": "cfg/tiny-yolo-voc-2c.cfg",
           "load": "bin/tiny-yolo-voc.weights",
           "batch": 16,
           "epoch": 500,
           "train": True,
           "annotation": "./train/annotation/",
           "dataset": "./train/images/"}

# Train the model
tfnet = TFNet(options)
tfnet.train()
# Built graph to a protobuf file (.pb)
tfnet.savepb()
