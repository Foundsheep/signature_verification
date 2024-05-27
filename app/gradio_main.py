import gradio as gr
from scripts.inference import inference

def greet(name):
    return f"Hello {name}!"


iface = gr.Interface(fn=inference, 
                     inputs=[gr.Image(), gr.Image()],
                     outputs=[gr.Textbox(), gr.Textbox()])

iface.launch(share=True)