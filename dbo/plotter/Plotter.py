from dbo.tester.Tester import *
from pandas import DataFrame
import pandas as pd

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,10)

class Plotter:
    def __init__(self, experiment_list : list[ExperimentResult], result_df : DataFrame):
        self.experiment_list = experiment_list
        self.result_df = result_df
    
    
    def plot_feature(self, name : str) -> None:
        functions = self.result_df['sim'].unique()
        acqf_func = self.result_df['acqf'].unique()

        fig, ax = plt.subplots(len(functions))
        fig.tight_layout(pad = 2.0)

        ind = 0
        for function in functions:
            for acqf in acqf_func:
                data = self.result_df[(self.result_df.sim == function)&(self.result_df.acqf == acqf)][name]

                ax[ind].hist(data, density = True,alpha = 0.5, label = acqf)

                ax[ind].set_title(function)
                ax[ind].legend()

            ind += 1
        
    def plot_min_sol(self, start_index : int  = 0):
        functions = self.result_df['sim'].unique()
        acqf_func = self.result_df['acqf'].unique()
        
        fig, ax = plt.subplots(len(functions))
        fig.tight_layout(pad = 2.0)
        color_list = ['blue', 'red', 'green', 'yellow']
        
        i = 0
        j = 0
        for function in functions:
            for acqf in acqf_func: 
                
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
                
                ax[i].plot(np.arange(start_index, length), mean, color = color_list[j%len(acqf_func)], label = acqf)
                ax[i].fill_between(np.arange(start_index, length), per_25, per_75, color = color_list[j%len(acqf_func)], alpha=.15)
                ax[i].legend()
                ax[i].set_title(function)
                j += 1
            i += 1
                
        
        