
import torch 
import torch.nn as nn 
import lightning as L 
import torch.nn.functional as F



class Skipgram(nn.Module):
    def __init__(self,vocab_size: int,embedding_dim: int):
        super(Skipgram,self).__init__()
        self.embeddings = nn.Embedding(vocab_size,embedding_dim)
        self.output = nn.Linear(embedding_dim,vocab_size)
        
        
    def forward(self,inputs):
        embeds = self.embeddings(inputs)
        output = self.output(embeds)
        return output 
    
    
    
    
    
    

class Word2vec(L.LightningModule):
    
    def __init__(self,vocab_size: int , embedding_dim: int , negative_sample:int = 5):
        
        super().__init__()
        self.model = Skipgram(vocab_size,embedding_dim) 
        self.vocab_size = vocab_size
        self.negative_sample = negative_sample
        
        
    def forward(self,x):
        return self.model(x)
    
    def training_step(self,batch, batch_idx):
        target,context = batch
        
                
        # Positive samples
        pos_loss = F.logsigmoid(torch.sum(self.forward(target) * self.model.embeddings(context), dim=1))
        
        # Negative samples
        neg_samples = torch.randint(0, self.vocab_size, (len(target), self.negative_samples))
        neg_loss = F.logsigmoid(-torch.sum(self.forward(target).unsqueeze(1) * self.model.embeddings(neg_samples), dim=2))
        
        loss = -(pos_loss + neg_loss.sum(1)).mean()
        
        self.log('train_loss', loss)
        return loss



    def configure_optimizers(self):
        optimiser = torch.optim.adam(self.parameters(),lr = 0.025)
        scheduler = torch.optim.lr_scheduler.StepLR(optimiser,step_size=1,gamma=0.9999)
        return [optimiser],[scheduler]