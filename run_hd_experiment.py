from dbo.simulator.TestFunctions import *
from dbo.acquisition.Wrappers import (EIWrapper, DEIWrapper, KGWrapper)
from dbo.tester.Tester import *
from dbo.metrics.ExperimentMetric import *
from dbo.plotter.Plotter import *
import pickle

test = Tester(5, 15, 10)
simulator_hd_list = [Hartmann6DSimulator(), Griewank5DSimulator(), Michalewicz5DSimulator()]
acquisition_list = [DEIWrapper(lambda_ = 1, epsilon = 0.1), EIWrapper()]

experiment_hd_list = test.perform_known_experiment(100, simulator_hd_list, acquisition_list)

pickle.dump(experiment_hd_list, open( "results/experiment_hd.pkl", "wb" ) )