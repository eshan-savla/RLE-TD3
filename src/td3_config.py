import hydra
from omegaconf import DictConfig, OmegaConf

hydra.initialize(config_path="../configs/", job_name="td3_config")
cfg = hydra.compose(config_name="config")