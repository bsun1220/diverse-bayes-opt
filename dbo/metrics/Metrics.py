from dbo.tester.Tester import *
import pandas as pd
from pandas import DataFrame

class Metrics:
    """
    Metrics Class - Used to generate data from ExperimentalResults
    
    """
    def __init__(self) -> None:
        pass
    
    def get_dataframe(self, result : list[ExperimentResult]) -> DataFrame:
        """
        return dataframe based on specified 

        Args
        ----------
        result : list[ExperimentalResult]
            result object after experiment test
        
        Returns 
        ----------
        Data : DataFrame
            Information DataFrame
        """
        return pd.DataFrame()