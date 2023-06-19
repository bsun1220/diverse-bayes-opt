from dbo.acquisition.AcquisitionWrapperClass import *
from dbo.acquisition.DEI import DiverseExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.acquisition import ExpectedImprovement, qKnowledgeGradient, PosteriorMean

class EIWrapper(AcquisitionWrapper):
    def __init__(self, maximize : bool = True, num_restarts : int = 5, raw_samples : int = 100):
        self.maximize = maximize
        super().__init__(num_restarts, raw_samples)

    def optimize_acquisition(self, best_f : float, model : FixedNoiseGP, model_bounds : Tensor) -> Tensor:
        acqf = ExpectedImprovement(model = model, best_f = best_f, maximize = self.maximize)
        new_point,_ = optimize_acqf(acqf, bounds=model_bounds, q = 1, num_restarts = self.num_restarts, raw_samples = self.raw_samples)
        return new_point

class DEIWrapper(AcquisitionWrapper):
    def __init__(self, lambda_ : float, epsilon : float, maximize : bool = False, num_restarts : int = 5, raw_samples : int = 100):
        assert lambda_ > 0
        assert epsilon > 0
        self.maximize = maximize
        super().__init__(num_restarts, raw_samples)
        self.lambda_ = lambda_
        self.epsilon = epsilon

    def optimize_acquisition(self, best_f : float, model : FixedNoiseGP, model_bounds : Tensor) -> Tensor:
        acqf = DiverseExpectedImprovement(model = model, best_f = best_f, lambda_ = self.lambda_, epsilon_ = self.epsilon)
        new_point,_ = optimize_acqf(acqf, bounds=model_bounds, q = 1, num_restarts = self.num_restarts, raw_samples = self.raw_samples)
        return new_point

class KGWrapper(AcquisitionWrapper):
    def __init__(self, num_fantasy : int, maximize : bool = False, num_restarts : int = 5, raw_samples : int = 100):
        self.num_fantasy = num_fantasy
        self.maximize = maximize
        super().__init__(num_restarts, raw_samples)

    def optimize_acquisition(self, best_f : float, model : FixedNoiseGP, model_bounds : Tensor) -> Tensor:

        qKG = qKnowledgeGradient(model, num_fantasies=self.num_fantasy)

        new_point,_ = optimize_acqf(qKG, bounds=model_bounds, 
                                  q = 1, num_restarts = self.num_restarts, raw_samples = self.raw_samples)
        return new_point