import tensorflow as tf
import pandas as pd
import numpy as np
from utils.model import ann
import logging
import os
import utils.model

logging_str = "[%(asctime)s %(levelname)s %(module)s] %(message)s"
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir,"runtime_logging.log"), level=logging.INFO, format=logging_str, filemode='a')

def main(layers,batchSize):
    logging.info(f"Loading mnist dataset...")
    (X_train_full, y_train_full),( X_test, y_test) = tf.keras.datasets.mnist.load_data()
    logging.info(f"mnist dataset loaded")
    logging.info(f"X_train_full shape: \n{X_train_full.shape}")
    logging.info(f"y_train_full shape: \n{y_train_full.shape}")
    logging.info(f"X_test shape: \n{X_test.shape}")
    logging.info(f"y_test shape: \n{y_test.shape}")
    print(f"mnist dataset loaded")
    print(f"X_train_full shape: \n{X_train_full.shape}")
    print(f"y_train_full shape: \n{y_train_full.shape}")
    print(f"X_test shape: \n{X_test.shape}")
    print(f"y_test shape: \n{y_test.shape}")
    X_valid = X_train_full[:5000]
    X_train = X_train_full[5000:]
    y_valid = y_train_full[:5000]
    y_train = y_train_full[5000:]
    logging.info(f"X_valid shape: \n{X_valid.shape}")
    logging.info(f"X_train shape: \n{X_train.shape}")
    logging.info(f"y_valid shape: \n{y_valid.shape}")
    logging.info(f"y_train shape: \n{y_train.shape}")
    print(f"X_valid shape: \n{X_valid.shape}")
    print(f"X_train shape: \n{X_train.shape}")
    print(f"y_valid shape: \n{y_valid.shape}")
    print(f"y_train shape: \n{y_train.shape}")
    ann_model = ann(layers)
    model = ann_model.get_model()
    model.compile(optimizer='sgd',loss='sparse_categorical_crossentropy', metrics='accuracy')
    model.fit(x=X_train,y=y_train,batch_size=32,epochs=1000,verbose=1)
            #,validation_data=(X_valid,y_valid))
    

if __name__ ==  '__main__':
    try:
        LAYERS = [
            tf.keras.layers.Flatten(input_shape=[28,28], name="inputLayer"),
            tf.keras.layers.Dense(300, activation='relu', name="hiddenLayer1"),
            tf.keras.layers.Dense(100, activation='relu', name="hiddenLayer2"),
            tf.keras.layers.Dense(10, activation='softmax', name="outputLayer")
        ]
        BATCH_SIZE = 32
        main(layers=LAYERS,batchSize=BATCH_SIZE)

    except Exception as e:
        logging.exception(f"Exception occured: \n{e}")
        raise e
