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

class DiverseExpectedImprovement(AnalyticAcquisitionFunction):
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
    """
    def __init__(self, model : GPyTorchModel, lambda_ : float, epsilon_ : float, best_f : float):
        super().__init__(model=model)
        self.register_buffer("lambda_", torch.as_tensor(lambda_))
        self.register_buffer("epsilon_", torch.as_tensor(epsilon_))
        self.register_buffer("best_f", torch.as_tensor(best_f))

    @t_batch_mode_transform(expected_q = 1)
    def forward(self, X: Tensor) -> Tensor:
        mean, sigma = self._mean_and_sigma(X)
        factor = (1 + self.epsilon_ ) if self.best_f < 0 else (1 - self.epsilon_)
        
        ei_portion = Phi((mean - self.best_f)/sigma) * (mean - self.best_f)
        dei_portion = phi((mean - self.best_f)/sigma) + self.lambda_ * Phi((mean - factor * self.best_f)/sigma)
        
        return ei_portion + dei_portion * sigma