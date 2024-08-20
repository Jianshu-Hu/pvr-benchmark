import os
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import gc
import sys
from tqdm import tqdm
import torch
from omegaconf import OmegaConf
import hydra
from PIL import Image

from vc_models import vc_models_dir_path


def create_data(config):
    # infer the demo location
    demo_paths_loc = os.path.join(config["dataset"]["data_dir"], config["dataset"]["env_name"] + ".pickle")
    try:
        demo_paths = pickle.load(open(demo_paths_loc, "rb"))
    except:
        print("Unable to load the data. Check the data path.")
        print(demo_paths_loc)
        quit()

    demo_paths = demo_paths[: config["dataset"]["num_demos"]]
    demo_score = np.mean([np.sum(p["rewards"]) for p in demo_paths])
    print("Number of demonstrations used : %i" % len(demo_paths))
    print("Demonstration score : %.2f " % demo_score)

    # compute embeddings and create dataset
    print("===================================================================")
    print(">>>>>>>>> Precomputing frozen embedding dataset >>>>>>>>>>>>>>>>>>>")
    demo_paths = compute_embeddings(
        demo_paths,
        device=config["device"],
        embedding_name=config["embedding_name"],
    )
    demo_paths = precompute_features(
        demo_paths,
        history_window=config["dataset"]["history_window"],
        fuse_embeddings=fuse_embeddings_flare,
        proprio_key=config["dataset"]["proprio_key"],
    )
    gc.collect()  # garbage collection to free up RAM
    dataset = DynamicsFrozenEmbeddingDataset(
        demo_paths,
        history_window=config["dataset"]["history_window"],
        fuse_embeddings=fuse_embeddings_flare,
    )
    # Dataset in this case is pre-loaded and on the RAM (CPU) and not on the disk
    dataloader = DataLoader(
        dataset,
        batch_size=config["dataset"]["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    return dataset, dataloader


class DynamicsFrozenEmbeddingDataset(Dataset):
    # this data set will return not only state and action, but also next state.
    def __init__(
        self,
        paths: list,
        history_window: int = 1,
        fuse_embeddings: callable = None,
        device: str = "cuda",
    ):
        self.paths = paths
        assert "embeddings" in self.paths[0].keys()
        # assume equal length trajectories
        # code will work even otherwise but may have some edge cases
        self.path_length = max([p["actions"].shape[0] for p in paths])
        self.num_paths = len(self.paths)
        self.history_window = history_window
        self.fuse_embeddings = fuse_embeddings
        self.device = device

    def __len__(self):
        return self.path_length * self.num_paths

    def __getitem__(self, index):
        traj_idx = int(index // self.path_length)
        timestep = int(index - traj_idx * self.path_length)
        timestep = min(timestep, self.paths[traj_idx]["actions"].shape[0]-2)
        if "features" in self.paths[traj_idx].keys():
            features = self.paths[traj_idx]["features"][timestep]
            action = self.paths[traj_idx]["actions"][timestep]
            next_features = self.paths[traj_idx]["features"][timestep+1]
        else:
            embeddings = [
                self.paths[traj_idx]["embeddings"][max(timestep - k, 0)]
                for k in range(self.history_window)
            ]
            embeddings = embeddings[
                ::-1
            ]  # embeddings[-1] should be most recent embedding
            features = self.fuse_embeddings(embeddings)
            # features = torch.from_numpy(features).float().to(self.device)
            action = self.paths[traj_idx]["actions"][timestep]
            # action   = torch.from_numpy(action).float().to(self.device)
            next_embeddings = [
                self.paths[traj_idx]["embeddings"][max(timestep+1 - k, 0)]
                for k in range(self.history_window)
            ]
            next_embeddings = next_embeddings[
                ::-1
            ]  # embeddings[-1] should be most recent embedding
            next_features = self.fuse_embeddings(next_embeddings)
        return {"features": features, "actions": action, "next_features": next_features}


def compute_embeddings(
    paths: list, embedding_name: str, device: str = "cpu", chunk_size: int = 20
):
    model, embedding_dim, transforms, metadata = load_pretrained_model(
        embedding_name=embedding_name
    )
    model.to(device)
    for path in tqdm(paths):
        inp = path["images"]  # shape (B, H, W, 3)
        path["embeddings"] = np.zeros((inp.shape[0], embedding_dim))
        path_len = inp.shape[0]
        preprocessed_inp = torch.cat(
            [transforms(frame) for frame in inp]
        )  # shape (B, 3, H, W)
        for chunk in range(path_len // chunk_size + 1):
            if chunk_size * chunk < path_len:
                with torch.no_grad():
                    inp_chunk = preprocessed_inp[
                        chunk_size * chunk : min(chunk_size * (chunk + 1), path_len)
                    ]
                    emb = model(inp_chunk.to(device))
                    # save embedding in RAM and free up GPU memory
                    emb = emb.to("cpu").data.numpy()
                path["embeddings"][
                    chunk_size * chunk : min(chunk_size * (chunk + 1), path_len)
                ] = emb
        del path["images"]  # no longer need the images, free up RAM
    return paths


def precompute_features(
    paths: list,
    history_window: int = 1,
    fuse_embeddings: callable = None,
    proprio_key: str = None,
):
    assert "embeddings" in paths[0].keys()
    for path in paths:
        features = []
        for t in range(path["embeddings"].shape[0]):
            emb_hist_t = [
                path["embeddings"][max(t - k, 0)] for k in range(history_window)
            ]
            emb_hist_t = emb_hist_t[
                ::-1
            ]  # emb_hist_t[-1] should correspond to time t embedding
            feat_t = fuse_embeddings(emb_hist_t)
            if proprio_key not in [None, "None"]:
                assert proprio_key in path["env_infos"].keys()
                feat_t = np.concatenate([feat_t, path["env_infos"][proprio_key][t]])
            features.append(feat_t.copy())
        path["features"] = np.array(features)
    return paths


# ===================================
# Model Loading
# ===================================
def load_pretrained_model(embedding_name, input_type=np.ndarray, *args, **kwargs):
    """
    Load the pretrained model based on the config corresponding to the embedding_name
    """

    config_path = os.path.join(
        vc_models_dir_path, "conf/model", embedding_name + ".yaml"
    )
    print("Loading config path: %s" % config_path)
    config = OmegaConf.load(config_path)
    model, embedding_dim, transforms, metadata = hydra.utils.call(config)
    model = model.eval()  # model loading API is unreliable, call eval to be double sure

    def final_transforms(transforms):
        if input_type == np.ndarray:
            return lambda input: transforms(Image.fromarray(input)).unsqueeze(0)
        else:
            return transforms

    return model, embedding_dim, final_transforms(transforms), metadata


# ===================================
# Temporal Embedding Fusion
# ===================================
def fuse_embeddings_concat(embeddings: list):
    assert type(embeddings[0]) == np.ndarray
    return np.array(embeddings).ravel()


def fuse_embeddings_flare(embeddings: list):
    if type(embeddings[0]) == np.ndarray:
        history_window = len(embeddings)
        delta = [embeddings[i + 1] - embeddings[i] for i in range(history_window - 1)]
        delta.append(embeddings[-1].copy())
        return np.array(delta).ravel()
    elif type(embeddings[0]) == torch.Tensor:
        history_window = len(embeddings)
        # each embedding will be (Batch, Dim)
        delta = [embeddings[i + 1] - embeddings[i] for i in range(history_window - 1)]
        delta.append(embeddings[-1])
        return torch.cat(delta, dim=1)
    else:
        print("Unsupported embedding format in fuse_embeddings_flare.")
        print("Provide either numpy.ndarray or torch.Tensor.")
        quit()