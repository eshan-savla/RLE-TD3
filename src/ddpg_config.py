import hydra
from omegaconf import DictConfig, OmegaConf

# Specify the config file name you want to use for the next training / evaluation run
# A list of all available config files can be found in the folder ./configs folder. 
# You can find a mapping of the existing models and configs in ./models/Mapping_mod-conf.md

config_name = "config" # Change the config name to your desired config file

with hydra.initialize(config_path="../configs/", job_name="ddpg_config"):
    cfg = hydra.compose(config_name=config_name)