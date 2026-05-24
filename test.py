import pickle
import time
import math
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.config import config
from src.model import Model
# from src.dataset import get_dataloader
from src.utils import save_checkpoint, load_checkpoint

OUTPUT_DIR = "/home/james/single"

# load the data
with open(f"{OUTPUT_DIR}/data/headings_data.pkl", "rb") as f:
    headings_data = pickle.load(f)

with open(f"{OUTPUT_DIR}/data/rel_positions_data.pkl", "rb") as f:
    rel_positions_data = pickle.load(f)

with open(f"{OUTPUT_DIR}/data/rel_traffic_data.pkl", "rb") as f:
    rel_traffic_data = pickle.load(f)


print(f"headings_data: {headings_data.shape}")
print(f"rel_positions_data: {rel_positions_data.shape}")
print(f"rel_traffic_data: {rel_traffic_data.shape}")


print("replacing inf with 0")


# replace the inf values with 0
replace_dict = {np.inf: 0}
for key, value in replace_dict.items():
    rel_positions_data[rel_positions_data == key] = value
    rel_traffic_data[rel_traffic_data == key] = value


rel_traffic_data = rel_traffic_data.squeeze(axis=2)


print(f"headings_data: {headings_data.shape}")
print(f"rel_positions_data: {rel_positions_data.shape}")
print(f"rel_traffic_data: {rel_traffic_data.shape}")




# torch.manual_seed(1)

# rel_traffic_data = rel_traffic_data.reshape(len(headings_data), config["context_length"]+1, 2*config["n_aircraft"])

# print(f"headings_data: {headings_data.shape}")
# print(f"rel_positions_data: {rel_positions_data.shape}")
# print(f"rel_traffic_data: {rel_traffic_data.shape}")



