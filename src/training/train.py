import lightning as L 
import torch
import torch.nn as nn 

from src.data.dataset import Word2vecDataModule
from src.model.word2vec import Word2VecLightningModule


def train_word2vec(file_path: str, embedding_dim: int = 300,batch_size: int = 64, max_epochs: int = 5):
    data_module = Word2vecDataModule(file_path,batch_size)
    data_module.setup()
    
    model = Word2VecLightningModule(data_module.vocab_size,embedding_dim)
    
    trainer = L.Trainer(max_epochs=max_epochs, gpus = 1 if torch.cuda.is_available() else None )
    trainer.fit(model,data_module)
    return model ,data_module



