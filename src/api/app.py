import fastapi
from fastapi import FastAPI

from src.training.train import train_word2vec

app = FastAPI()

@app.post("/train")
async def train_model(file_path: str, embedding_dim:int = 300 , batch_size : int = 64,max_epochs : int = 5):
    model,data_module = train_word2vec(file_path,embedding_dim,batch_size,max_epochs)
    return  {"message": "model Trained Succesfully"}

