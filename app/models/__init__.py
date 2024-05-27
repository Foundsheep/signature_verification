import torch
from .siamese_network import SiameseNetwork
from ..configs import *

model = SiameseNetwork()
if DEVICE == "cuda":
    model.load_state_dict(torch.load(WEIGHT_PATH))
else:
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location="cpu"))