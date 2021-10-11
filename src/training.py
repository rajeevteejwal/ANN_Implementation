from src.utils.data_mgmt import get_data
from src.utils.model import create_model
from src.utils.common import read_config
import tensorflow as tf
import argparse


def training(config_path):
    config = read_config(config_path)

    validation_data_size = config["params"]["validation_datasize"]
    (X_train, y_train), (X_valid, y_valid), (X_test,y_test) = get_data(validation_data_size)
    loss_function = config["params"]["loss_function"]
    optimizer = config["params"]["optimizer"]
    metrics = config["params"]["metrics"]
    batch_size = config["params"]["batch_size"]
    epochs = config["params"]["epochs"]
    is_tensorboard_logging_enable = config["tensorboard"]["is_tensorboard_logging_enable"]
    tensorboard_logs_dir = config["tensorboard"]["logs_dir"]
    # NUM_CLASSES = config["params"]["num_classes"]
    # tensorboard logging enabled
    callbacks = []
    if is_tensorboard_logging_enable:
        tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_logs_dir)
        callbacks.append(tensorboard_cb)
        # tb_early_stopping = tf.keras

    model = create_model(loss_function=loss_function, metrics=metrics, optimizer=optimizer)

    model.fit(x=X_train, y=y_train, epochs=epochs,
              validation_data=(X_valid, y_valid), batch_size=batch_size, callbacks=callbacks)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="config.yaml")
    parsed_args = args.parse_args()
    print(f"Training: \n{parsed_args}")
    print(f"Training: config \n{parsed_args.config}")
    training(config_path=parsed_args.config)
