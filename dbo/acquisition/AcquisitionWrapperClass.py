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


class AcquisitionWrapper():

    def __init__(self, num_restarts : int = 5, raw_samples : int = 100):
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples

    def optimize_acquisition(self, best_f : float, model : FixedNoiseGP, model_bounds : Tensor) -> Tensor:
        return torch.rand(1)
        
