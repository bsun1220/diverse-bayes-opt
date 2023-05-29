from dbo.metrics.Metrics import *
import pandas as pd
from pandas import DataFrame
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class ExperimentMetrics(Metrics):
    def __init__(self):
        super().__init__()
    
    
    def get_sil_score(self, points, max_k = 5):
        if points.shape[0] < 6:
            return 1

        sil = []
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters = k).fit(points)
            labels = kmeans.labels_
            sil.append(silhouette_score(points, labels, metric = 'euclidean'))

        return np.argmax(np.array(sil)) + 2
        
    def get_dataframe(self, experiment_list : list[ExperimentResult], eps : float) -> DataFrame:
        columns = ["sim", "acqf", "num_sol", "curr_min", "avg_dist", "num_cluster", "trial"]
        result = pd.DataFrame(columns = columns)
        
        for experiment in experiment_list:
            min_val = experiment.y.min()
            train_x = experiment.x
            
            factor = (1 + eps) if experiment.min > 0 else (1 - eps)
            
            torch_ind = experiment.y < factor * experiment.min
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
    
    
            res = [experiment.sim, experiment.acqf, len(feasible_x), min_val.item(), avg_dist, cluster_num, experiment.trial]
            result.loc[len(result)] = res
        
        
        return result