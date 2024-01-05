import hydra
from omegaconf import DictConfig, OmegaConf

config_name = "config_6"
with hydra.initialize(config_path="../configs/", job_name="td3_config"):
    cfg = hydra.compose(config_name=config_name) # Option to test multiple configurations: Change the config name to your desired config file