import torch
from pathlib import Path
import albumentations as A

BATCH_SIZE=8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: [{DEVICE}]")
TRIPLET_LOSS_MARGIN = 50.0
MODEL = "SiameseNetwork"
LOSS = "BCELoss"
EPOCHS = 2

# 상대경로로 하려고 했으나, WEIGHT_PATH를 쓰는 곳과 여기의 경로가 다름...
WEIGHT_PATH = "/code/app/models/checkpoints/20240523_135239/epoch_0039.pt"
# WEIGHT_PATH = "/code/app/models/checkpoints/20240524_172730/epoch_0066.pt"
# WEIGHT_PATH = "/code/app/models/checkpoints/20240524_172730/epoch_0056.pt"
# WEIGHT_PATH = "/code/app/models/checkpoints/20240524_172730/epoch_0035.pt"

if not Path(WEIGHT_PATH).exists:
    print(f"[{WEIGHT_PATH}] weight_path not exists")

USE_PRE_TRAINED = True

# TODO: 도커 내에서 접근할 수 없음. 접근 방법 필요
ROOT_DIR = r"C:\Users\msi\Desktop\workspace\015_twin_networks\02_resources\245.개인 특정을 위한 자필과 모사 필기체 데이터\01-1.정식개방데이터"
# ROOT_DIR = "/root/workspace/01_tn/resources/245.개인_특정을_위한_자필과_모사_필기체_데이터/01-1.정식개방데이터"
# transform
TRANSFORM = A.Compose(
    [
        A.Resize(height=88, width=765),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),        
    ]
)