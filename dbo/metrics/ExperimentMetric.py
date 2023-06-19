from dbo.metrics.Metrics import *
import pandas as pd
from pandas import DataFrame
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from torch import Tensor

class ExperimentMetrics(Metrics):
    """
    ExperimentMetrics Class used after testing TestFunctions
    
    """
    def __init__(self):
        super().__init__()
    
    
    def get_sil_score(self, points : Tensor, max_k : int = 5) -> int:
        """
        Get most indicative cluster number in solution set

        Args
        ----------
        points : Tensor
            list of points to perform clustering on
        
        max_k : int
            maximum cluster number to test
        
        Returns 
        ----------
        Index : int
            Best fit clustern number after using silhouette score
        """
        if points.shape[0] < 6:
            return 1

        sil = []
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters = k).fit(points)
            labels = kmeans.labels_
            try: 
                sil.append(silhouette_score(points, labels, metric = 'euclidean'))
            except:
                sil.append(0)

        return np.argmax(np.array(sil)) + 2
        
    def get_dataframe(self, experiment_list : list[ExperimentResult], eps : float) -> DataFrame:
        """
        Return dataframe based on specified 

        Args
        ----------
        experiment_list : list[ExperimentalResult]
            result object after experiment test
        eps : float
            error parameter characterizing what is 
            in the solution set
        
        Returns 
        ----------
        Data : DataFrame
            DataFrame giving current minimum, average distance in solution set, 
            number of clusters in solution found by KMeans, and trial number
        """
        columns = ["sim", "acqf", "num_sol", "curr_max", "avg_dist", "num_cluster", "trial"]
        result = pd.DataFrame(columns = columns)
        
        for experiment in experiment_list:
            max_val = experiment.y.max()
            
            factor = (1 + eps) if experiment.max < 0 else (1 - eps)
            threshold = factor * experiment.max if experiment.max != 0 else -eps
            
            torch_ind = experiment.y > threshold
            ind = []
            i = 0
            for val in torch_ind:
                if val:
                    ind.append(i)
                i += 1
            
            feasible_sol, feasible_x = experiment.y[ind], experiment.x[ind]
            avg_dist = 0

            ind = 0
            for i in range(feasible_sol.shape[0]):
                for j in range(i + 1, feasible_sol.shape[0]):
                    val1, val2 = feasible_x[i], feasible_x[j]
                    avg_dist += torch.norm(val1 - val2).item()
                    ind += 1

            avg_dist = 0 if ind == 0 else avg_dist/ind
            
            cluster_num = self.get_sil_score(feasible_x)
    
            res = [experiment.sim, experiment.acqf, len(feasible_x), max_val.item(), avg_dist, cluster_num, experiment.trial]
            result.loc[len(result)] = res
        
        return result