from fastapi import FastAPI
import gradio as gr
from .scripts.inference import inference


app = FastAPI()


@app.get("/")
def hello_world():
    return {"message": "OK"}

@app.post("/train")
def train():
    return {"message": "OK"}

GRADIO_PATH = "/gradio"

io = gr.Interface(fn=inference, 
                  inputs=[gr.Image(), gr.Image()],
                  outputs=[gr.Textbox(), gr.Textbox()])

app = gr.mount_gradio_app(app, io, path=GRADIO_PATH)
