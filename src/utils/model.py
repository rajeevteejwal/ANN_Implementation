import tensorflow as tf
# import logging


def create_model(loss_function, metrics, optimizer):
    layers = [
            tf.keras.layers.Flatten(input_shape=[28, 28], name="inputLayer"),
            tf.keras.layers.Dense(300, activation='relu', name="hiddenLayer1"),
            tf.keras.layers.Dense(100, activation='relu', name="hiddenLayer2"),
            tf.keras.layers.Dense(10, activation='softmax', name="outputLayer")
        ]

    # logging.info(f"Creating model of layers: \n{layers}")
    print(f"Creating model of layers: \n{layers}")
    model = tf.keras.Sequential(layers)
    # logging.info(f"Model created with following configuration: \n{model.summary()}")
    print(f"Model summary: \n{model.summary()}")

    model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)

    return model
