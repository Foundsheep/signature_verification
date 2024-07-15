import torch
from .siamese_network import SiameseNetwork
from configs import *
import traceback

try:
    model = SiameseNetwork()
    if DEVICE == "cuda":
        model.load_state_dict(torch.load(WEIGHT_PATH))
    else:
        model.load_state_dict(torch.load(WEIGHT_PATH, map_location="cpu"))
except Exception as e:
    model = None
    print(e)
    traceback.print_exc()