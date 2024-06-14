# custom handler file

# model_handler.py

"""
ModelHandler defines a custom model handler.
"""

import base64
import io
import torch
import albumentations as A

from PIL import Image
from ts.torch_handler.base_handler import BaseHandler

class ModelHandler(BaseHandler):
    """
    A custom model handler implementation.
    """

    def __init__(self):
        self._context = None
        self.initialized = False
        self.explain = False
        self.target = 0
        self.image_processing = A.Compose(
            [
                A.Resize(height=88, width=765),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),        
            ]
        )

        

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        self._context = context
        self.initialized = True
        #  load the model, refer 'custom handler class' above for details

        properties = context.system_properties
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")
        
    def preprocess(self, data):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        # Take the input data and make it inference ready

        received = data[0].get("data")
        if received is None:
            received = data[0].get("body")

        print(f"recieved : {received}")
        img_1, img_2 = received["img_1"], received["img_2"]

        # type check
        if type(img_1) != type(img_2):
            raise TypeError(f"img_1 and img_2 have different data types. {type(img_1) = }, {type(img_2) = }")

        # preprocess
        if isinstance(img_1, str):
            print("...image is string!")
            img_1 = base64.b64decode(img_1)
            img_2 = base64.b64decode(img_2)
        elif isinstance(img_1, (bytearray, bytes)):
            print("...image bytesarray!")
            img_1 = Image.open(io.BytesIO(img_1))
            img_2 = Image.open(io.BytesIO(img_2))
        else:
            print(f"...image type: [{type(img_1)}]")
            # if the image is a list
            img_1 = torch.FloatTensor(img_1)
            img_2 = torch.FloatTensor(img_2)

        img_1 = self.image_processing(image=img_1)["image"]
        img_2 = self.image_processing(image=img_2)["image"]

        return img_1.to(self.device), img_2.to(self.device)

    def inference(self, img_1, img_2):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """
        # Do some inference call to engine here and return output
        model_output = self.model.forward(img_1, img_2)
        return model_output

    def postprocess(self, inference_output):
        """
        Return inference result.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        # Take output from network and post-process to desired format
        postprocess_output = inference_output
        postprocess_output = postprocess_output.item()
        return postprocess_output

    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
        model_input = self.preprocess(data)
        model_output = self.inference(model_input)
        return self.postprocess(model_output)