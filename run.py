import dotenv
import hydra
from omegaconf import DictConfig

from src.predict import predict

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):
    if config.mode == "train":
        # Imports should be nested inside @hydra.main to optimize tab completion
        # Read more here: https://github.com/facebookresearch/hydra/issues/934
        from src.train import train
        from src.utils import utils

        # A couple of optional utilities:
        # - disabling python warnings
        # - easier access to debug mode
        # - forcing debug friendly configuration
        # You can safely get rid of this line if you don't want those
        utils.extras(config)

        # Pretty print config using Rich library
        if config.get("print_config"):
            utils.print_config(config, resolve=True)

        # Train model
        return train(config)
    elif config.mode == "predict":
        predict(config)
    else:
        raise Exception(f"given mode ({config.mode}) is not recognized.")


if __name__ == "__main__":
    main()
