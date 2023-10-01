from dbo.acquisition.AcquisitionWrapperClass import *
from dbo.acquisition.DEI import DiverseExpectedImprovement
from dbo.acquisition.CDEI import ContourDiverseImprovement
from dbo.acquisition.NormDEI import *
from botorch.optim import optimize_acqf
from botorch.acquisition import ExpectedImprovement, qKnowledgeGradient, UpperConfidenceBound

class EIWrapper(AcquisitionWrapper):
    def __init__(self, num_restarts : int = 5, raw_samples : int = 100):
        super().__init__(num_restarts, raw_samples)

    def optimize_acquisition(self, model_input : Tensor, model_output: Tensor, model : FixedNoiseGP, model_bounds : Tensor) -> Tensor:
        best_f = model_output.min()
        acqf = ExpectedImprovement(model = model, best_f = best_f, maximize = False)
        new_point,_ = optimize_acqf(acqf, bounds=model_bounds, q = 1, num_restarts = self.num_restarts, raw_samples = self.raw_samples)
        return new_point

class DEIWrapper(AcquisitionWrapper):
    def __init__(self, lambda_ : float, epsilon : float, num_restarts : int = 5, raw_samples : int = 100):
        assert lambda_ > 0
        assert epsilon > 0
        super().__init__(num_restarts, raw_samples)
        self.lambda_ = lambda_
        self.epsilon = epsilon

    def optimize_acquisition(self, model_input : Tensor, model_output: Tensor, model : FixedNoiseGP, model_bounds : Tensor) -> Tensor:
        best_f = model_output.min()
        acqf = DiverseExpectedImprovement(model = model, best_f = best_f, lambda_ = self.lambda_, epsilon_ = self.epsilon)
        new_point,_ = optimize_acqf(acqf, bounds=model_bounds, q = 1, num_restarts = self.num_restarts, raw_samples = self.raw_samples)
        return new_point
    
class NormDEIWrapper(AcquisitionWrapper):
    def __init__(self, epsilon : float, norm : Norm, num_restarts : int = 5, raw_samples : int = 100):
        assert epsilon > 0
        super().__init__(num_restarts, raw_samples)
        self.epsilon = epsilon
        self.norm = norm

    def optimize_acquisition(self,model_input : Tensor, model_output: Tensor, model : FixedNoiseGP, model_bounds : Tensor) -> Tensor:
        best_f = model_output.min()
        best_x = model_input[model_output.argmin().item()]
        
        acqf = AltDiverseImprovement(model = model, epsilon_ = self.epsilon, best_f = best_f, best_x = best_x, norm = self.norm)
        new_point,_ = optimize_acqf(acqf, bounds=model_bounds, q = 1, num_restarts = self.num_restarts, 
                                    raw_samples = self.raw_samples, sequential = True)
        return new_point
    
class CDEIWrapper(AcquisitionWrapper):
    def __init__(self, lambda_ : float, epsilon : float, UCB : bool, num_restarts : int = 5, raw_samples : int = 100):
        assert lambda_ > 0
        assert epsilon > 0
        super().__init__(num_restarts, raw_samples)
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.UCB = UCBS

    def optimize_acquisition(self, model_input : Tensor, model_output: Tensor, model : FixedNoiseGP, model_bounds : Tensor) -> Tensor:
        best_f = None
        if not self.UCB:
            best_f = model_output.min()
        else:
            LCB = UpperConfidenceBound(model, beta=0.2, maximize = False)
            point, _ = optimize_acqf(LCB, bounds=model_bounds, q = 1, 
                                     num_restarts = self.num_restarts//2, raw_samples = self.raw_samples//2)
            best_f = LCB(point).item()
        
        acqf = ContourDiverseImprovement(model = model, best_f = best_f, lambda_ = self.lambda_, epsilon_ = self.epsilon)
        new_point,_ = optimize_acqf(acqf, bounds=model_bounds, q = 1, num_restarts = self.num_restarts, raw_samples = self.raw_samples)
        return new_point


class KGWrapper(AcquisitionWrapper):
    def __init__(self, num_fantasy : int, num_restarts : int = 5, raw_samples : int = 100):
        self.num_fantasy = num_fantasy
        super().__init__(num_restarts, raw_samples)

    def optimize_acquisition(self, model_input : Tensor, model_output: Tensor, model : FixedNoiseGP, model_bounds : Tensor) -> Tensor:

        qKG = qKnowledgeGradient(model, num_fantasies=self.num_fantasy)

        new_point,_ = optimize_acqf(qKG, bounds=model_bounds, 
                                  q = 1, num_restarts = self.num_restarts, raw_samples = self.raw_samples)
        return new_point
        
    
        