import tensorflow as tf
import numpy as np
import pandas as pd
import logging

class ann:
    def __init__(self, layers):
        self.layers = layers
        logging.info(f"ANN class variable is initialized with layers details:\n{self.layers}")

    def get_model(self):
        logging.info(f"Creating model of layers: \n{self.layers}")
        print(f"Creating model of layers: \n{self.layers}")
        model = tf.keras.Sequential(self.layers)
        logging.info(f"Model created with following configuration: \n{model.summary()}")
        print(f"Model summary: \n{model.summary()}")
        return model
    