import torch

class Config():
    BATCH_SIZE=8
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: [{DEVICE}]")
    TRIPLET_LOSS_MARGIN = 50.0
    MODEL = "SiameseNetwork"
    LOSS = "TripletMarginLoss"
    EPOCHS = 50
