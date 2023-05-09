from abc import ABC, abstractmethod
import numpy as np
import random
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Net(ABC):

    def __init__(self, seed=42):
        #set_seed(seed)
        pass

    @abstractmethod
    def predict(self, x):
        pass
    
    @abstractmethod
    def learn(self, data):
        pass