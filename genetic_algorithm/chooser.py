import numpy as np


class DefaultChooser:
    def __init__(self, replacement: bool = False) -> None:
        self.replace = replacement
    def __call__(self, individuals, weights):
        if weights==0:
            return np.random.choice(individuals, 2, self.replace)
        return np.random.choice(individuals, 2, self.replace, p=weights)
    
    def prepare(self, individuals, weights):
        sum_ = np.sum(weights)
        if sum_ == 0:
            weights = 0
        else:
            weights = weights/sum_
            if len(weights[weights==2]) < 2:
                weights = 0
        return individuals, weights

class TopChooser:
    def __init__(self, top: int, replace: bool = False) -> None:
        self.top = top
        self.replace = replace

    def __call__(self, individuals, weights):
        if self.top > len(individuals):
            self.top = len(individuals)
        if weights==0:
            return np.random.choice(individuals[:self.top], 2, self.replace)
        return np.random.choice(individuals[:self.top], 2, self.replace, p=weights)
    
    def prepare(self, individuals, weights):
        sum_ = np.sum(weights[:self.top])
        if sum_ == 0:
            weights = 0
        else:
            weights = weights[:self.top]/sum_
            if len(weights[weights==2]) < 2:
                weights = 0
        return individuals[:self.top], weights