import torch
from torch import Tensor
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.utils.transforms import t_batch_mode_transform
from botorch.utils.probability.utils import (
    ndtr as Phi,
    phi,
)
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.gp_regression import FixedNoiseGP

class ContourDiverseImprovement(AnalyticAcquisitionFunction):
    """
    Diverse Expected Improvement Implementation
    
    Args
    ----------
    model : GPyTorchModel
        Gaussian Process Model to be used
    lambda_ : float
        Exploration hyperparameter and tuner
    epsilon_ : float
        Threshold for error
    best_f : float
        Current best maximum found so far
    best_x : Tensor
        Current x of the best maximum found so far
    """
    def __init__(self, model : GPyTorchModel, lambda_ : float, epsilon_ : float, best_f : float):
        super().__init__(model=model)
        self.register_buffer("lambda_", torch.as_tensor(lambda_))
        self.register_buffer("epsilon_", torch.as_tensor(epsilon_))
        self.register_buffer("best_f", torch.as_tensor(best_f))

    @t_batch_mode_transform(expected_q = 1)
    def forward(self, X: Tensor) -> Tensor:
        mean, sigma = self._mean_and_sigma(X)
        gamma = self.best_f + self.epsilon_
        z = (gamma - mean)/sigma
        
        a = (gamma - mean) * sigma * (phi(z) - 3 * phi(z + self.lambda_))
        b = ((mean - gamma)**2) * (2 * Phi(z) - Phi(z + self.lambda_))
        c = self.lambda_ * phi(z + self.lambda_) + 2 * Phi(z) + ((self.lambda_**2) -1) * Phi(z + self.lambda_)
        c *= (sigma ** 2)
        
        return a + b + c