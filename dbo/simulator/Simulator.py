import numpy as np
import torch
from torch import Tensor
import collections

class Simulator:
    """
    General simulator wrapper class
    
    Args
    ----------
    dim : float
        Dimension of the problem
    uid : str
        Unique ID for identification purposes
    bounds : Tensor
        Tensor object relating bounds for the problem
    """
    def __init__(self, dim:int, uid:str, bounds:Tensor) -> None:
        assert(bounds.shape[1] == dim)
        
        self.dim = dim
        self.uid = uid
        self.bounds = bounds
    
    def generate(self, X:Tensor) -> Tensor:
        """
        For a given input, generate the corresponding output from the simulator

        Args
        ----------
        X : Tensor
            input which matches dimension and bounds criteria
        
        Returns
        ----------
        Tensor : output of the simulation
        
        """
        return 0
    
    def diversity_metric(self, X:Tensor, Y:Tensor) -> float:
        """
        Specify a current diversity metric between two points

        Args
        ----------
        X : Tensor
            input 1
        Y : Tensor
            input 2
        
        Returns
        ----------
        float : a diversity metric 
        """
        return 0
    
    def scalarize_input(self, train_x : Tensor) -> Tensor:
        """
        Convert input of any dimension into [0,1] hypercube

        Args
        ----------
        train_x : Tensor
            Training X value to be converted
        
        Returns
        ----------
        Tensor : converted value
        """
        train_x_i = torch.clone(train_x)
        for dim in range(self.bounds.shape[1]):
            bound = self.bounds[:, dim]
            train_x_i[:, dim] -= bound[0]
            train_x_i[:, dim] /= ((bound[1] - bound[0]))
        return train_x_i
    
    def revert_input(self, train_x_i : Tensor) -> Tensor:
        """
        Convert [0,1] input back into unnormalized form

        Args
        ----------
        train_x : Tensor
            Training X value to be converted
        
        Returns
        ----------
        Tensor : converted value
        """
        train_x = torch.clone(train_x_i)
        for dim in range(self.bounds.shape[1]):
            bound = self.bounds[:, dim]
            train_x[:, dim] *= ((bound[1] - bound[0]))
            train_x[:, dim] += bound[0]
        return train_x
    
    def scalarize_output(self, train_obj : Tensor) -> tuple[Tensor, float, float]:
        """
        Convert output into z-score normalized

        Args
        ----------
        train_obj : Tensor
            Training objective value to be converted
        
        Returns
        ----------
        Tensor : converted value
        """
        mean = train_obj.mean().item()
        std = train_obj.std().item()
        return (train_obj - mean)/std, mean, std
    
    def revert_output(self, train_obj : Tensor, mean : float, std : float) -> Tensor:
        """
        Convert output from z-score normalized into regular form

        Args
        ----------
        train_obj : Tensor
            Training objective value to be converted
        
        Returns
        ----------
        Tensor : converted value
        """
        assert std > 0
        return train_obj * std + mean