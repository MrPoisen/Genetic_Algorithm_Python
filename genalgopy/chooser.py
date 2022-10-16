from abc import abstractmethod, ABC
from typing import List
import numpy as np

from .genetic import Individual

class AbstractChooser(ABC):
    @abstractmethod
    def __call__(self, individuals: List[Individual], weights: List[float]):
        pass

    @abstractmethod
    def prepare(self, individuals: List[Individual], weights: List[float]):
        pass


class DefaultChooser(AbstractChooser):
    def __init__(self, replace: bool = False) -> None:
        """
        Args:
            replace: should be False if the choosen samples should be unique
        """
        self.replace = replace
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

class TopChooser(AbstractChooser):
    def __init__(self, top: int, replace: bool = False) -> None:
        """
        Args:
            top: how many of the best Individuals should be choosen from
            replacement: should be False if the choosen samples should be unique
        """
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