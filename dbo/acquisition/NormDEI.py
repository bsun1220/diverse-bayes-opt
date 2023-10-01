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
from dbo.distance.Distance import Norm

class NormDiverseImprovement(AnalyticAcquisitionFunction):
    """
    Alternative Diverse Improvement Implementation
    
    Args
    ----------
    model : GPyTorchModel
        Gaussian Process Model to be used
    epsilon_ : float
        Threshold for error
    best_f : float
        Current best maximum found so far
    best_x : Tensor
        Current x found for the best max
    norm : Norm
        Norm class which determines diversity between 2 tensors
    """
    def __init__(self, model : GPyTorchModel, epsilon_ : float, best_f : float, best_x : Tensor, norm : Norm):
        super().__init__(model=model)
        self.register_buffer("epsilon_", torch.as_tensor(epsilon_))
        self.register_buffer("best_f", torch.as_tensor(best_f))
        self.register_buffer("best_x", best_x)
        self.norm = norm

    @t_batch_mode_transform(expected_q = 1)
    def forward(self, X: Tensor) -> Tensor:

        mean, sigma = self._mean_and_sigma(X)
        factor = (1 - self.epsilon_ ) if self.best_f < 0 else (1 + self.epsilon_)
        z = (self.best_f - mean)/sigma
        exploit = Phi(z) * (self.best_f - mean)
        diverse = self.norm.norm(self.best_s, X) * (Phi((factor * self.best_f - mean)/sigma) - Phi(z))
        explore = sigma * (phi(z) + diverse)
        
        return exploit + explore