import torch
from PIL import Image
import albumentations as A
import numpy as np


from ..models import model
from ..configs import *

model.to(DEVICE)
print(f"model to [{DEVICE}]")

def read_image_for_inference(image_path):
    # TODO: configs.py에 넣기
    def _transform_image(image):
        transform = A.Compose(
            [
                A.Resize(height=88, width=765),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),        
            ]
        )

        image = transform(image=image)["image"]
        return image
    
    # get numpy array image
    # -- 1. from path
    if isinstance(image_path, str):
        image = Image.open(image_path)  # there was a case when 150th image was not valid
        image = np.array(image)[:, :, :3]  # image has alpha channel -> 4 channels
    # -- 2. from np.ndarray
    elif isinstance(image_path, np.ndarray):
        image = image_path
    else:
        raise ValueError(f"not expected type of argument. image_path : [{type(image_path)}]")
    
    try:
        # transform
        image = _transform_image(image)
        
        # convert to torch.tensor
        image = image.transpose((2, 0, 1))
        image = torch.tensor(image, dtype=torch.float)

        # batch dimension added
        image = image.to(DEVICE).unsqueeze(dim=0)
    except:
        image = None
    return image


def inference(input_1_path, input_2_path):
    # print(f"input_1_path : [{type(input_1_path)}], input_2_path : [{type(input_2_path)}]")

    if isinstance(input_1_path, str) and isinstance(input_2_path, str):
        img_1 = read_image_for_inference(input_1_path)
        img_2 = read_image_for_inference(input_2_path)
    elif isinstance(input_1_path, np.ndarray) and isinstance(input_2_path, np.ndarray):
        img_1 = read_image_for_inference(input_1_path)
        img_2 = read_image_for_inference(input_2_path)
    elif isinstance(input_1_path, Image.Image) and isinstance(input_2_path, Image.Image):
        img_1 = input_1_path
        img_2 = input_2_path
    else:
        raise ValueError(f"arguments' type is wrong. input_1_path : [{type(input_1_path)}], input_2_path : [{type(input_2_path)}]")
    
    result = None
    prob = None
    try:
        model.eval()

        with torch.no_grad():
            outs = model(img_1, img_2)
            pred = torch.where(outs > 0.5, 1, 0)
        
        if pred == 1:
            result = "similar"
        else:
            result = "dissimilar"
        prob = outs.item()
        prob = f"{prob :.4f}"
    except:
        result = "Error occured"
        prob = "Erro occured"
    return result, prob