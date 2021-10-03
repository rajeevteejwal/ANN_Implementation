from src.training import training
import argparse

def main(config_path):
    training(config_path)

if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config.yaml")
    parsed_args = args.parse_args()
    print(f"Training: \n{parsed_args}")
    print(f"Training: config \n{parsed_args.config}")
    training(config_path=parsed_args.config)
