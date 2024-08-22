import torch
import torch.nn as nn
import utils


class Fusion(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super().__init__()
        self.out_dim = out_dim
        layers = [nn.Linear(input_dim, hidden_dim[0]), nn.ReLU(inplace=True)]
        for ind in range(0, len(hidden_dim) - 1):
            layers.append(nn.Linear(hidden_dim[ind], hidden_dim[ind+1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(hidden_dim[-1], out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class ForwardDynamics(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim[0]), nn.ReLU(inplace=True)]
        for ind in range(0, len(hidden_dim) - 1):
            layers.append(nn.Linear(hidden_dim[ind], hidden_dim[ind+1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(hidden_dim[-1], out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim[0]), nn.ReLU(inplace=True)]
        for ind in range(0, len(hidden_dim) - 1):
            layers.append(nn.Linear(hidden_dim[ind], hidden_dim[ind+1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(hidden_dim[-1], out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class DynamicsModule(nn.Module):
    def __init__(self, input_dim, fuse_hidden_dim, feature_dim, action_dim, dynamics_hidden_dim):
        super().__init__()
        self.fusion = Fusion(input_dim, fuse_hidden_dim, feature_dim)
        self.feature_dim = feature_dim
        # ema
        self.target_fusion_tau = 0.01
        self.target_fusion = Fusion(input_dim, fuse_hidden_dim, feature_dim)

        self.dynamics = ForwardDynamics(feature_dim+action_dim, dynamics_hidden_dim, feature_dim)

        self.decoder = Decoder(feature_dim, dynamics_hidden_dim, input_dim)

        self.apply(utils.weight_init)

        self.target_fusion.load_state_dict(self.fusion.state_dict())
        self.target_fusion.eval()

    def forward(self, obs, action, next_obs):
        feature = self.fusion(obs)
        decoded_feature = self.decoder(feature)
        cat_feature = torch.cat([feature, action], dim=-1)
        predicted_target = self.dynamics(cat_feature)
        with torch.no_grad():
            target = self.target_fusion(next_obs)
            # target = next_obs
        return target, predicted_target, decoded_feature
