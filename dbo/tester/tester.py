from dbo.simulator.Simulator import *
from dbo.acquisition import *
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from collections import namedtuple
import torch
from botorch.optim import optimize_acqf
from torch import Tensor
from botorch.models.gp_regression import FixedNoiseGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
import warnings
warnings.filterwarnings("ignore")

BayesOptResult = namedtuple("BayesOptResult", ['x', 'y'])
ExperimentResult = namedtuple("ExperimentResult", ['trial', 'acqf', 'sim', 'x', 'y', 'min'])

class Tester:
    """
    Tester Experimentation Implementation

    Args
    ----------
    num_design : int
        Number of design points per dimension
    num_sim : int
        Number of optimization points per dimension
    refit_param : int
        Refitting rate per dimension
    error_term : float
        error term in fixed noise Gaussian Process
    """
    def __init__(self, num_design : int, num_sim : int, refit_param : 
                 int, error_term : float = 10**(-2)) -> None:
        self.num_design = num_design
        self.num_sim = num_sim
        self.refit_param = refit_param
        self.error_term = error_term
    
    def perform_bayes_opt(self, simulator : Simulator, 
                          acquisition : AnalyticAcquisitionFunction,
                          param_settings : dict = {}) -> BayesOptResult:
        """
        Full bayesian optimization loop

        Args
        ----------
        simulator : Simulator
            a function to optimize over
        acquisition : AnalyticAcquisitionFunction
            a custom acquisition function 
        param_settings : dict
            custom parameter settings for the acquisition function
        
        Returns
        ----------
        result : BayesOptResult
            Returns all x and y values from the optimization loop
        """
        dim = simulator.dim
        
        #actual observed
        obs_x = simulator.revert_input(torch.rand(self.num_design * dim, dim))
        obs_obj = simulator.generate(obs_x).unsqueeze(-1)
        
        #regularized inputs to model
        model_bounds = torch.tensor([[0.0] * dim, [1.0] * dim])
        model_input = simulator.scalarize_input(obs_x)
        model_output, mean, sigma = simulator.scalarize_output(obs_obj)
        var = torch.zeros(model_output.shape) + self.error_term
        
        model = FixedNoiseGP(train_X = model_input, train_Y = model_output, train_Yvar = var)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        
        for i in range(self.num_sim * dim):
            best_f = model_output.min()
            acqf = acquisition(model = model, best_f = model_output.min(), **param_settings)
            new_point, _ = optimize_acqf(acqf, bounds=model_bounds, q = 1, num_restarts = 5, raw_samples = 100)
            eval_point = simulator.revert_input(new_point)
            eval_result = simulator.generate(eval_point).expand(1, 1)
            
            reg_eval_result = (eval_result - mean)/sigma
            model_input = torch.cat((model_input, new_point), 0)
            model_output = torch.cat((model_output, reg_eval_result), 0)
            var = torch.cat((var, torch.as_tensor(self.error_term).expand(1,1)), 0)
            model = model.condition_on_observations(X = new_point, Y = reg_eval_result, 
                                                noise = torch.as_tensor(self.error_term).expand(1,1))
            
            if i % dim * self.refit_param == 0:
                model_output = simulator.revert_output(model_output, mean, sigma)
                model_output, mean, sigma = simulator.scalarize_output(model_output)
                model = FixedNoiseGP(train_X = model_input, train_Y = model_output, train_Yvar = var)
                mll = ExactMarginalLogLikelihood(model.likelihood, model)
                fit_gpytorch_mll(mll)
        
        obs_x = simulator.revert_input(model_input)
        obs_obj = simulator.revert_output(model_output, mean, sigma)
        
        return BayesOptResult(obs_x, obs_obj)
            
        
    
    def perform_known_experiment(self, num_trials : int, simulator_list: list[Simulator],
                           acquisition_list : list[tuple[AnalyticAcquisitionFunction, dict]]) -> list[ExperimentResult]:
        """
        Performs optimization over trial, acquisition, and simulator

        Args
        ----------
        num_trials : int
            Number of total trials
        simulator_list : list[Simulator]
            list offunctions to optimize over
        acquisition_list : list[tuple[AnalyticAcquisitionFunction, dict]]
            a custom acquisition function list with parameter settings
        
        Returns
        ----------
        result : list[ExperimentResult]
            list of result containing, trial, acqf, sim, x, y, and min
        """
        ans_list = []
        
        for trial in range(num_trials):
            for simulator in simulator_list:
                
                for acquisition_detail in acquisition_list:
                    acquisition, acquisition_params = acquisition_detail
  
                    result = self.perform_bayes_opt(simulator, acquisition, acquisition_params)
                    
                    exp_result = ExperimentResult(trial + 1, acquisition.__name__, 
                                                  simulator.__class__.__name__, result.x, result.y, simulator.true_min)

                    ans_list.append(exp_result)
        
        return ans_list
            
        
        