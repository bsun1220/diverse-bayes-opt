from dbo.simulator.Simulator import *
from dbo.acquisition import *

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
    """
    def __init__(self, num_design : int, num_sim : int, refit_param : int) -> None:
        self.num_design = num_design
        self.num_sim = num_sim
        self.refit_param = refit_param
    
    def perform_bayes_opt(self, simulator : Simulator, acquisition : AnalyticAcquisitionFunction) -> Tensor:
        pass
    
    def perform_experiment(self, num_trials : int, 
                           simulator_list: list[Simulator], acquisition_list : list[AnalyticAcquisitionFunction]):
        
        pass