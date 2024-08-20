import pytorch_lightning as pl
from torch import optim
import importlib
from omegaconf import OmegaConf
import argparse
import torch.nn as nn

import sys
sys.path.append('../')
from utils import soft_update_params
from dataset import create_data


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


class TrainingModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        # initialize dynamics module
        self.dynamics_module = instantiate_from_config(config.model)

    def training_step(self, batch, batch_idx):
        current_feature, action, next_feature = batch['features'].float(), batch['actions'].float(),\
                                                batch['next_features'].float()

        target, predicted_target = self.dynamics_module(current_feature, action, next_feature)
        loss = nn.functional.mse_loss(target, predicted_target)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)

        # update ema
        soft_update_params(self.dynamics_module.fusion, self.dynamics_module.target_fusion,
                           self.dynamics_module.target_fusion_tau)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(list(self.dynamics_module.fusion.parameters()) +
                               list(self.dynamics_module.dynamics.parameters()), lr=1e-3)
        return optimizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="cfgs/config.yaml", help="config file")
    parser.add_argument("--name", type=str, default="test")
    parser.add_argument("--embedding", type=str, default="dinov2_base_patch14")
    parser.add_argument("--env", type=str, default="dmc_walker_stand-v1")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    config.env = args.env
    config.embedding_name = args.embedding
    print(config)
    pl.seed_everything(args.seed)

    # -------------------------dataset-------------------------
    # preprocess the image with encoder and build the dataset
    dataset, dataloader = create_data(config)
    # set feature dim and action dim for the dynamics model
    one_sample = dataset[0]
    config.model.params.input_dim = one_sample['features'].shape[-1]
    config.model.params.action_dim = one_sample['actions'].shape[-1]
    print("===================================================================")
    print(">>>>>>>>> Finishing building dataset >>>>>>>>>>>>>>>>>>>")

    # -------------------------dynamics model-------------------------
    training_module = TrainingModule(config)
    config.lightning.trainer.default_root_dir = config.lightning.trainer.default_root_dir+'/'+args.name
    trainer = pl.Trainer(**config.lightning.trainer)

    # -------------------------train-------------------------
    print("===================================================================")
    print(">>>>>>>>> Start training >>>>>>>>>>>>>>>>>>>")
    trainer.fit(model=training_module, train_dataloaders=dataloader)


if __name__ == '__main__':
    main()