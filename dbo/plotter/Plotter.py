from dbo.tester.Tester import *
from dbo.metrics.ExperimentMetric import *
from pandas import DataFrame
import pandas as pd

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,10)

class Plotter:
    """
    Plotter tool to visualize results of experimental run

    Parameters
    ----------
    experiment_list : list[ExperimentResult]
        full results object
    eps : float
        penalization error for what is considered a feasible set
    """
    def __init__(self, experiment_list : list[ExperimentResult], eps : float = 0.3) -> None:
        self.experiment_list = experiment_list

        experiment_metric = ExperimentMetrics()
        self.result_df = experiment_metric.get_dataframe(experiment_list, eps)
        self.eps = eps
    
    
    def plot_feature(self, name : str) -> None:
        """
        Plot the histogram for a specific feature

        Parameters
        ----------
        name : str
            feature name
        """
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
        
    def plot_min_sol(self, start_index : int  = 0) -> None:
        """
        Plot the existing minimum solution over trials

        Parameters
        ----------
        start_index : int
            starting index in the minimum solution chart
        """
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
        
    def plot_scatter_2d(self, trial : int) -> None:
        """
        Plot scatter 2d over input values

        Parameters
        ----------
        trial : int
            trial number to look over
        """
        functions = self.result_df['sim'].unique()
        acqf_func = self.result_df['acqf'].unique()
        fig, ax = plt.subplots(len(functions), len(acqf_func))
        fig.tight_layout(pad = 2.0)

        i = 0
        j = 0

        for function in functions:
            for acqf in acqf_func:

                experiment_one = None
                for experiment in self.experiment_list:
                    if experiment.trial == trial and experiment.sim == function and experiment.acqf == acqf:
                        experiment_one = experiment
                        break

                min_val = experiment_one.y.min()


                factor = (1 + self.eps) if experiment_one.min > 0 else (1 - self.eps)

                torch_ind = experiment_one.y < factor * experiment_one.min
                ind = []
                k = 0
                for val in torch_ind:
                    if val:
                        ind.append(k)
                    k += 1

                feasible_sol, feasible_x = experiment_one.y[ind], experiment_one.x[ind]

                ax[i][j%len(acqf_func)].scatter(experiment_one.x[:, 0],experiment_one.x[:, 1], label = "Solutions")
                ax[i][j%len(acqf_func)].scatter(feasible_x[:, 0],feasible_x[:, 1], label = "Feasible", color = "red")
                ax[i][j%len(acqf_func)].set_title(function + " "+ acqf)
                ax[i][j%len(acqf_func)].legend()

                j += 1

            i += 1

    def plot_local_minima(self, local_minima : dict) -> None:
        """
        Plot local minima per specified points

        Parameters
        ----------
        local_minima : dict
            each function contains a certain number of points
        """
        functions = self.result_df['sim'].unique()
        acqf_func = self.result_df['acqf'].unique()

        trial_num = self.result_df['trial'].max()

        i = 0
        j = 0

        fig, ax = plt.subplots(len(functions), len(acqf_func))
        fig.tight_layout(pad = 2.0)

        for function in functions:
            minima_list = local_minima[function]
            for acqf in acqf_func:

                res = np.zeros((trial_num, len(minima_list)))

                ind = 0
                for experiment in self.experiment_list:

                    if experiment.sim != function or experiment.acqf != acqf:
                        continue

                    for k in range(len(minima_list)):
                        min_dist = float("inf")
                        for row in experiment.x.numpy():
                            min_dist = min(min_dist, np.linalg.norm(minima_list[k] - row))
                        res[ind][k] = min_dist
                    ind += 1

                xs = ['x' + str(k) for k in range(1, len(minima_list) + 1)]
                ax[i][j % len(acqf_func)].bar(xs, res.mean(axis = 0))
                ax[i][j % len(acqf_func)].set_title(function + " " + acqf)
                j += 1
            i += 1
                
                
                        
                        
                                        
                            
                        
                    
                    
                
                
        
        
                    
                    
                    
                    
                    
        
        