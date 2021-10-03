from src.utils.data_mgmt import get_data
from src.utils.model import create_model
from src.utils.common import read_config
import argparse

def training(config_path):
    config = read_config(config_path)

    validation_datasize = config["params"]["validation_datasize"]
    (X_train, y_train), (X_valid, y_valid), (X_test,y_test) = get_data(validation_datasize)

    LOSS_FUNCTION = config["params"]["loss_function"]
    OPTIMIZER = config["params"]["optimizer"]
    METRICS = config["params"]["metrics"]
    BATCH_SIZE = config["params"]["batch_size"]
    EPOCHS = config["params"]["epochs"]
    #NUM_CLASSES = config["params"]["num_classes"]

    model = create_model(loss_function=LOSS_FUNCTION, metrics=METRICS, optimizer=OPTIMIZER)

    model.fit(x=X_train, y=y_train,epochs=EPOCHS, validation_data=(X_valid,y_valid), batch_size=BATCH_SIZE)

if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config.yaml")
    parsed_args = args.parse_args()
    print(f"Training: \n{parsed_args}")
    print(f"Training: config \n{parsed_args.config}")
    training(config_path=parsed_args.config)
