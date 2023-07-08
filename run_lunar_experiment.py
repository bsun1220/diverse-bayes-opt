from dbo.simulator.TestFunctions import *
from dbo.acquisition.Wrappers import (EIWrapper, DEIWrapper, KGWrapper)
from dbo.tester.Tester import *
from dbo.metrics.ExperimentMetric import *
from dbo.plotter.SimPlotter import *
from dbo.simulator.LunarLanding.LunarSimulator import LunarSimulator
import pickle

test = Tester(5, 15, 10)
simulator_list = [LunarSimulator()]
acquisition_list = [DEIWrapper(lambda_ = 1, epsilon = 0.05), EIWrapper()]
experiment_list = test.perform_known_experiment(100, simulator_list, acquisition_list)

pickle.dump(experiment_list, open( "results/experiment_lunar.pkl", "wb" ) )