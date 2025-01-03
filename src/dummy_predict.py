from fastapi import FastAPI


app = FastAPI()

@app.get("/")
def ping():
    return {"message": "pong"}
