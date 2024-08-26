from fastapi import FastAPI, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from typing import Annotated
import traceback
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).absolute().parent))
from app.scripts.inference import inference
app = FastAPI()

origins = [
    "*",
]

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def hello_world():
    return {"message": "OK"}

@app.post("/inference")
def infer(
    input_1: Annotated[UploadFile, Form()],
    input_2: Annotated[UploadFile, Form()]
):
    result = None
    prob = None
    try:
        result, prob = inference(input_1=input_1,
                                 input_2=input_2)
    except Exception as e:
        print(e)
        traceback.print_exc()
    return {"result": result,
            "prob": prob}

