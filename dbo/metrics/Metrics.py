from dbo.tester.Tester import *
import pandas as pd
from pandas import DataFrame

class Metrics:
    def __init__(self):
        pass
    
    def get_dataframe(self, result : list[ExperimentResult]) -> DataFrame:
        return pd.DataFrame()