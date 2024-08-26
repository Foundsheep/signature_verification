import torch
from PIL import Image
import io
import albumentations as A
import numpy as np
from pathlib import Path
from starlette.datastructures import UploadFile # used instead of from fastapi import UploadFile

from models import model
from configs import *

model.to(DEVICE)
print(f"model to [{DEVICE}]")

def read_image_for_inference(input_):
    image = None
    # get numpy array image
    # -- 1. from path
    if isinstance(input_, str):
        if Path(input_).exists():
            image = Image.open(input_)  # there was a case when 150th image was not valid
            image = np.array(image)
        else:
            print(f"input type is string, but seems like not a path")
    # -- 2. from np.ndarray
    elif isinstance(input_, np.ndarray):
        image = input_
    elif isinstance(input_, Image.Image):
        image = np.array(image)
    elif isinstance(input_, UploadFile):
        image = np.array(Image.open(io.BytesIO(input_.file.read())))
    else:
        raise ValueError(f"not expected type of argument. input_ : [{type(input_)}]")
    
    # if image has alpha channel -> 4 channels
    image = image[:, :, :3]
    try:
        # transform
        image = TRANSFORM(image=image)["image"]
        
        # convert to torch.tensor
        image = image.transpose((2, 0, 1))
        image = torch.tensor(image, dtype=torch.float)

        # batch dimension added
        image = image.to(DEVICE).unsqueeze(dim=0)
    except:
        image = None
    return image


def inference(input_1, input_2):
    img_1 = read_image_for_inference(input_1)
    img_2 = read_image_for_inference(input_2)

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