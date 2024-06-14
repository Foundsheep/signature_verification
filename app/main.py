from fastapi import FastAPI



app = FastAPI()


@app.get("/")
def hello_world():
    return {"message": "OK"}

@app.post("/train")
def train():
    return {"message": "OK"}

@app.post("/inference")
def train():

    return {"message": "OK"}

