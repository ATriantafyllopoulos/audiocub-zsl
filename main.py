import hydra

from omegaconf import (
    DictConfig, 
    OmegaConf
)


from train import main

@hydra.main(config_path="configs", config_name="config")
def experiment(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    main(cfg)

if __name__ == "__main__":
    experiment()