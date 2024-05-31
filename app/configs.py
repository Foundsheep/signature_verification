import torch
from pathlib import Path
import albumentations as A
import os

BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: [{DEVICE}]")
TRIPLET_LOSS_MARGIN = 50.0
MODEL = "SiameseNetwork_OutputEmbedding"
LOSS = "BCELoss"
EPOCHS = 50


WEIGHT_PATH = os.path.join(os.path.dirname(__file__), "models/checkpoints/20240523_135239/epoch_0039.pt")
# WEIGHT_PATH = "/code/app/models/checkpoints/20240524_172730/epoch_0066.pt"
# WEIGHT_PATH = "/code/app/models/checkpoints/20240524_172730/epoch_0056.pt"
# WEIGHT_PATH = "/code/app/models/checkpoints/20240524_172730/epoch_0035.pt"

if not Path(WEIGHT_PATH).exists:
    print(f"[{WEIGHT_PATH}] weight_path not exists")

USE_PRE_TRAINED = True

ROOT_DIR = "/root/workspace/01_twin_network/01_resources/245.개인 특정을 위한 자필과 모사 필기체 데이터/01-1.정식개방데이터"

# transform
TRANSFORM = A.Compose(
    [
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ]
)