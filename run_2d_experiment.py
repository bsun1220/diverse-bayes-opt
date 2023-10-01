from dbo.simulator.TestFunctions import *
from dbo.acquisition.Wrappers import (EIWrapper, DEIWrapper, AltDEIWrapper, CDEIWrapper)
from dbo.tester.Tester import *
from dbo.metrics.ExperimentMetric import *
from dbo.plotter.Plotter import *
from dbo.distance.Distance import EuclideanNorm
import pickle

norm = EuclideanNorm(2)

test = Tester(5, 15, 10)
simulator_2d_list = [Branin2DSimulator(), Griewank2DSimulator(), 
                  SixHumpCamel2DSimulator(), HolderTable2DSimulator(),  Gramacy2DSimulator()]
acquisition_list = [CDEIWrapper(1, 0.1, True), DEIWrapper(lambda_ = 1, epsilon = 0.1), 
                    EIWrapper(), AltDEIWrapper(epsilon = 0.1, norm = norm)]
experiment_2d_list = test.perform_known_experiment(100, simulator_2d_list, acquisition_list)
pickle.dump(experiment_2d_list, open( "results/experiment_2d.pkl", "wb" ) )