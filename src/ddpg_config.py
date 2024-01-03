import hydra
from omegaconf import DictConfig, OmegaConf
config_name = "config"
with hydra.initialize(config_path="../configs/", job_name="ddpg_config"):
    cfg = hydra.compose(config_name=config_name)