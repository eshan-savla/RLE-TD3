import hydra
from omegaconf import DictConfig, OmegaConf

with hydra.initialize(config_path="../configs/", job_name="td3_config"):
    cfg = hydra.compose(config_name="config") # Option to test multiple configurations: Change the config name to your desired config file