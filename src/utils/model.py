import tensorflow as tf
#import logging


def create_model(loss_function, metrics, optimizer):
    LAYERS = [
            tf.keras.layers.Flatten(input_shape=[28,28], name="inputLayer"),
            tf.keras.layers.Dense(300, activation='relu', name="hiddenLayer1"),
            tf.keras.layers.Dense(100, activation='relu', name="hiddenLayer2"),
            tf.keras.layers.Dense(10, activation='softmax', name="outputLayer")
        ]
    LOSS_FUNCTION = loss_function
    METRICS = metrics
    OPTIMIZER = optimizer
    #logging.info(f"Creating model of layers: \n{LAYERS}")
    print(f"Creating model of layers: \n{LAYERS}")
    model = tf.keras.Sequential(LAYERS)
    #logging.info(f"Model created with following configuration: \n{model.summary()}")
    print(f"Model summary: \n{model.summary()}")

    model.compile(optimizer=OPTIMIZER,loss=LOSS_FUNCTION, metrics=METRICS)

    return model