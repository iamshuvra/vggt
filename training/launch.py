# at the top of vggt/training/launch.py

import os
from omegaconf import DictConfig
from hydra import main as hydra_main
from vggt.training.trainer import Trainer

# single-GPU defaults
os.environ["LOCAL_RANK"]  = os.environ.get("LOCAL_RANK",  "0")
os.environ["RANK"]        = os.environ.get("RANK",        "0")
os.environ["WORLD_SIZE"]  = os.environ.get("WORLD_SIZE",  "1")
os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "127.0.0.1")
os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")

@hydra_main(
    version_base=None,
    config_path="config",   
    config_name="default",
)
def launch(cfg: DictConfig) -> None:
    Trainer(**cfg).run()

if __name__ == "__main__":
    launch()




















# # Copyright (c) Meta Platforms, Inc. and affiliates.
# # All rights reserved.
# #
# # This source code is licensed under the license found in the
# # LICENSE file in the root directory of this source tree.

# import os, pathlib
# os.environ.setdefault("LOCAL_RANK", "0") 
# os.environ.setdefault("RANK", "0") 
# os.environ.setdefault("WORLD_SIZE", "1") 
# os.environ.setdefault("MASTER_ADDR", "127.0.0.1") 
# os.environ.setdefault("MASTER_PORT", "29500")
# CONFIG_DIR = pathlib.Path(__file__).parent / "config"


# import argparse
# from hydra import initialize, compose
# from omegaconf import DictConfig, OmegaConf
# from vggt.training.trainer import Trainer



# def main():
#     parser = argparse.ArgumentParser(description="Train model with configurable YAML file")
#     parser.add_argument(
#         "--config", 
#         type=str, 
#         default="default",
#         help="Name of the config file (without .yaml extension, default: default)"
#     )
#     args, overrides = parser.parse_known_args()


#     with initialize(version_base=None, config_path="config"):
#         cfg = compose(config_name=args.config, overrides=overrides)

#     trainer = Trainer(**cfg)
#     trainer.run()


# if __name__ == "__main__":
#     main()


