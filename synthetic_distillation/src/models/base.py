import torch.nn as nn

class BasePolicy(nn.Module):
    def act(self, state, deterministic=False):
        pass

    def evaluate_actions(self, states, actions):
        pass

class DiscretePolicy(BasePolicy):
    pass
    
class ContinuousPolicy(BasePolicy):
    pass
