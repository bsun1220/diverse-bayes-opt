from dbo.simulator.LunarLanding.LunarSimulator import LunarSimulator
from dbo.acquisition.Wrappers import (EIWrapper, DEIWrapper, AltDEIWrapper, CDEIWrapper)
from dbo.tester.Tester import *
from dbo.metrics.ExperimentMetric import *
from dbo.plotter.Plotter import *
from dbo.distance.Distance import EuclideanNorm
import pickle

norm = EuclideanNorm(2)

test = Tester(5, 15, 10)
simulator_list = [LunarSimulator()]
acquisition_list = [CDEIWrapper(1, 0.5, True), DEIWrapper(lambda_ = 1, epsilon = 0.1), 
                    EIWrapper(), AltDEIWrapper(epsilon = 0.1, norm = norm)]
experiment_list = test.perform_known_experiment(100, simulator_list, acquisition_list)
pickle.dump(experiment_list, open( "results/experiment_lunar.pkl", "wb" ) )