from dbo.simulator.Simulator import *
from botorch.test_functions import Branin, Griewank, SixHumpCamel, HolderTable, Hartmann, Michalewicz

class Branin2DSimulator(Simulator):
    def __init__(self, uid : str = ''):
        bounds = torch.tensor([[-5.0, 0.0], [10.0, 15.0]])
        
        super().__init__(2, uid, bounds)
        self.true_min = 0.397887
        self.Branin = Branin()
    
    def generate(self, X:Tensor) -> Tensor:
        return self.Branin(X)

class Griewank2DSimulator(Simulator):
    def __init__(self, uid : str = ''):
        bounds = torch.tensor([[-5.0] * 2, [5.0] * 2])
        
        super().__init__(2, uid, bounds)
        self.true_min = 0
        self.Griewank = Griewank()
    
    def generate(self, X:Tensor) -> Tensor:
        return self.Griewank(X)

class SixHumpCamel2DSimulator(Simulator):
    def __init__(self, uid : str = ''):
        bounds = torch.tensor([[-3.0, -2.0], [3.0, 2.0]])
        
        super().__init__(2, uid, bounds)
        self.true_min = -1.0316
        self.SixHumpCamel = SixHumpCamel()
    
    def generate(self, X:Tensor) -> Tensor:
        return self.SixHumpCamel(X)

class HolderTable2DSimulator(Simulator):
    def __init__(self, uid : str = ''):
        bounds = torch.tensor([[-10, -10.0], [10.0, 10.0]])
        
        super().__init__(2, uid, bounds)
        self.true_min = -19.2
        self.HolderTable = HolderTable()
    
    def generate(self, X:Tensor) -> Tensor:
        return self.HolderTable(X)

class Gramacy2DSimulator(Simulator):
    def __init__(self, uid : str = ''):
        bounds = torch.tensor([[-2.0, -2.0], [6.0, 6.0]])
        
        super().__init__(2, uid, bounds)
        self.true_min = -0.5
    
    def generate(self, x:Tensor) -> Tensor:
        if len(x.shape) == 1:
            return torch.tensor(-np.abs(x[0] * np.exp(-(x[0]**2) - (x[1]**2))))
    
        lst = [] 
        for i in range(x.shape[0]):
            val = -np.abs(x[i][0] * np.exp(-(x[i][0]**2) - (x[i][1]**2)))
            lst.append(val)
        return torch.tensor(lst)
    
class Hartmann6DSimulator(Simulator):
    def __init__(self, uid : str = ''):
        bounds = torch.tensor([[0.0] * 6, [1.0] * 6])
        
        super().__init__(6, uid, bounds)
        self.true_min = -3.32237
        self.Hartmann = Hartmann(dim = 6)
    
    def generate(self, X:Tensor) -> Tensor:
        return self.Hartmann(X)

class Griewank5DSimulator(Simulator):
    def __init__(self, uid : str = ''):
        bounds = torch.tensor([[-5.0] * 5, [5.0] * 5])
        
        super().__init__(5, uid, bounds)
        self.true_min = 0
        self.Griewank = Griewank(dim = 5)
    
    def generate(self, X:Tensor) -> Tensor:
        return self.Griewank(X)

class Michalewicz5DSimulator(Simulator):
    def __init__(self, uid : str = ''):
        bounds = torch.tensor([[0.0] * 5, [3.14] * 5])
        
        super().__init__(5, uid, bounds)
        self.true_min = -4.687
        self.Michalewicz = Michalewicz(dim = 5)
    
    def generate(self, X:Tensor) -> Tensor:
        return self.Michalewicz(X)