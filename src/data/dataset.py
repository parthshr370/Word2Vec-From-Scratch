# import the damn libraries 

import torch.nn as nn
import lightning as L
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np 
import re # this is for regular expression 
from typing import List, Tuple
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS



# creating the corpus of text - class corpus


class Corpus:
    def __init__(self,file_path:str):
        self.file_path = file_path
        self.words = self.read_corpus()
        self.word_counts = Counter(self.words)
        self.vocab = self.build_vocab()
        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)} # dictionary mapping from word to inext 
        self.idx2word = {idx: word for word, idx in self.word2idx.items()} # dictionary mapping index to word for retreival when training testing 
        
        
    def read_corpus(self) -> List[str]:
        with open(self.file_path,'r',encoding= 'utf-7') as f:
            return [word.lower() for line in f for word in re.findall(r'\w+', line) ]
        
    def build_vocab(self,min_count: int = 5) -> List[str]:
        return [word for word, count in self.word_counts.items() if count >= min_count]
    
    
    


## Dataset class - that tokenises the words and all and then cretae vectors and index 

class Word2vecDataset(Dataset):
    def __init__(self,corpus: Corpus , window_size: int=5):
        self.corpus  = corpus
        self.window_size = window_size
        self.data = self.create_dataset() # we will define this function later on 
        
    def create_dataset(self) -> List[tuple[int,int]]:
        data = [] # define the vector that contains the dataset 
        for i , target in enumerate(self.corpus.words):
            target_idx = self.corpus.word2idx.get(target)
            if target_idx is None:
                continue
            context_words = self.corpus.words[max(0, i - self.window_size):i] + \
                            self.corpus.words[i + 1:i + 1 + self.window_size]
            for context in context_words:
                context_idx = self.corpus.word2idx.get(context)
                if context_idx is not None:
                    data.append((target_idx, context_idx))
        return data
    
    def __len__(self): # Returns the total number of samples
        return len(self.data)
    
    def __getitem__(self, idx): # returns the returns a specific sample in pytorch
        return torch.tensor(self.data[idx][0]), torch.tensor(self.data[idx][1]) # returns two rensors with index 0 and 1




# Lightning Data Module 


class Word2vecDataModule(L.LightningDataModule):
    def __init__(self, file_path: str, batch_size : int = 64):
        super().__init__()
        self.file_path = file_path
        self.batch_size = batch_size
        
    def setup(self, stage= None): # initializes the necessary components for training a Word2Vec model using text data from a specified file path
        corpus = Corpus(self.file_path)
        self.datataset = Word2vecDataset(corpus)
        self.vocab_size = len(corpus.vocab)
        
        
    def train_dataloader(self) :
        return DataLoader(self.dataset, batch_size=self.batch_size,shuffle=True,num_workers=4)
    
    
    