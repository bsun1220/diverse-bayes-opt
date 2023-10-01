import torch
import numpy as np

class Norm:
    def __init__(self):
        return
    
    def norm(self, x1 : torch.Tensor, x2 : torch.Tensor) -> float:
        return torch.linalg.norm(x1 - x2, dim=0).item()
    
class EuclideanNorm(Norm):
    def __init__(self, p : int):
        self.p = p
        
    def norm(self, x1 : torch.Tensor, x2 : torch.Tensor) -> float:
        d = x1.size()[0]
        return torch.linalg.norm(x1 - x2[0][0], dim=0, ord = self.p).item()/np.sqrt(d)
        