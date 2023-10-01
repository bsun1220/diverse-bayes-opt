from dbo.tester.Tester import *
from dbo.metrics.ExperimentMetric import *
from pandas import DataFrame
import pandas as pd

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,10)

class SimPlotter:
    """
    Plotter tool to visualize results of experimental run

    Parameters
    ----------
    experiment_list : list[ExperimentResult]
        full results object
    """
    def __init__(self, experiment_list : list[ExperimentResult]) -> None:
        self.experiment_list = experiment_list

        self.sim_list = set()
        self.acqf_list = set()
        for experiment in experiment_list:
            self.sim_list.add(experiment.sim)
            self.acqf_list.add(experiment.acqf)
    
    def plot_min(self, start_index : int  = 0) -> None:
        """
        Plot the optimization gap

        Parameters
        ----------
        start_index : int
            starting index in the optimization_gap
        """
        
        fig, ax = plt.subplots(max(len(self.sim_list), 2))
        fig.tight_layout(pad = 2.0)
        fig.suptitle("Optimization Gap")
        color_list = ['blue', 'red', 'green', 'yellow']

        i = 0
        j = 0
        for function in self.sim_list:
            for acqf in self.acqf_list: 
                
                lst = []
                for experiment in self.experiment_list:
                    if experiment.sim == function and experiment.acqf == acqf:
                        lst.append(experiment)
                
                data = np.zeros((len(lst), len(lst[0].x)))
                
                ind = 0
                length = len(data[0])
                for experiment in lst:
                    val = np.minimum.accumulate(experiment.y.numpy()).flatten()
                    data[ind] = val
                    ind += 1
                
                data = data[:, start_index:]
                mean = np.median(data, axis = 0)
                per_25 = np.quantile(data, q = 0.25, axis = 0)
                per_75 = np.quantile(data, q = 0.75, axis = 0)
                
                ax[i].plot(np.arange(start_index, length), mean, color = color_list[j%len(self.acqf_list)], label = acqf)
                ax[i].fill_between(np.arange(start_index, length), per_25, per_75, color = color_list[j%len(self.acqf_list)], alpha=.15)
                ax[i].legend()
                ax[i].set_title(function)
                j += 1
            i += 1

        plt.subplots_adjust(top=0.925)