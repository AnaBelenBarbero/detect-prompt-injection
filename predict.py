import pickle
from fastapi import FastAPI
def load_model():
    with open("model.pkl", "rb") as r:
        model = pickle.load(r)
        return model
    
#TODO: Add data model interface

app = FastAPI()

@app.get("/")
def ping():
    return {"message": "pong"}

@app.get("/predict")
def predict():
    model = load_model()
    return {"classes": model.classes_.tolist()}