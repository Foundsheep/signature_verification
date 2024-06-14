import torch
from .siamese_network import SiameseNetwork, SiameseNetwork_OutputEmbedding
from configs import *

# get a model
if LOSS == "BCELoss":
    model = SiameseNetwork()
elif LOSS == "TripletMarginWithDistanceLoss":
    model = SiameseNetwork_OutputEmbedding()

# load weights
if DEVICE == "cuda":
    model.load_state_dict(torch.load(WEIGHT_PATH))
else:
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location="cpu"))