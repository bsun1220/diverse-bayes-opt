from dbo.simulator.Simulator import *
from dbo.simulator.LunarLanding.lunar_landing_utils import *

class LunarSimulator(Simulator):
    def __init__(self, uid : str = '', seed : int = 1220):
        bounds = torch.tensor([[0.0] * 12, [1.0] * 12])
        self.seed = seed
        dim = 12
        self.true_min = 'unknown'
        super().__init__(dim, uid, bounds)

    def generate(self, X : Tensor) -> Tensor:
        X = X.numpy()
        res = np.zeros(len(X))

        ind = 0
        for row in X:
            reward = simulate_lunar_lander((row, self.seed))
            res[ind] = -reward
            ind += 1

        return torch.Tensor(res)
        
        
        