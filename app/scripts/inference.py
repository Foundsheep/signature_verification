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
        image = _transform_image(image)
        
        # convert to torch.tensor
        image = image.transpose((2, 0, 1))
        image = torch.tensor(image, dtype=torch.float)

        # batch dimension added
        image = image.to(DEVICE).unsqueeze(dim=0)
    except:
        image = None
    return image


def inference(input_1, input_2):
    # print(f"input_1 : [{type(input_1)}], input_2 : [{type(input_2)}]")

    # if isinstance(input_1, bytes) and isinstance(input_2, bytes):
    #     img_1 = 
    #     img_1 = Image.open(io.BytesIO(input_2.file.read()))
    # elif isinstance(input_1, str) and isinstance(input_2, str):
    #     img_1 = read_image_for_inference(input_1)
    #     img_2 = read_image_for_inference(input_2)
    # elif isinstance(input_1, np.ndarray) and isinstance(input_2, np.ndarray):
    #     img_1 = read_image_for_inference(input_1)
    #     img_2 = read_image_for_inference(input_2)
    # elif isinstance(input_1, Image.Image) and isinstance(input_2, Image.Image):
    #     img_1 = input_1
    #     img_2 = input_2
    # else:
    #     raise ValueError(f"arguments' type is wrong. input_1 : [{type(input_1)}], input_2 : [{type(input_2)}]")

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