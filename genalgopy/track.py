from abc import ABC, abstractmethod
from statistics import mean, median
from typing import List, Tuple

from .genetic import Individual

class AbstractTracker(ABC):
    @abstractmethod
    def __call__(self, fitness: List[Tuple[float, Individual]], force: bool = False):
        pass

    @property
    @abstractmethod
    def total_best(self) -> Tuple[int, float, Individual]:
        pass
    
    @property
    @abstractmethod
    def best_individual(self) -> Individual:
        pass
    
    @abstractmethod
    def stats_fromgen(self, start_gen: int, stop_gen: int) -> str:
        pass

    @abstractmethod
    def full_stats(self) -> str:
        pass


class Tracker(AbstractTracker):
    def __init__(self, frequency=1) -> None:
        """
        Args:
            frequency: how often information shall be gathered (ex.: frequency=2 means every second generation)"""
        self.best_per_gen = {}
        self.avg_per_gen = {}
        self.median_per_gen = {}
        self.gen = 0
        self.frq = frequency

    def __call__(self, fitness: List[Tuple[float, Individual]], force: bool = False):
        """
        Args:
            fitness: List of individuals and their respective fitness score
            force: if True, will ignore the frequency for this call            
        """
        self.gen += 1
        if (self.gen % self.frq == 0) or force:
            self.best_per_gen[self.gen] = fitness[0]
            fitnesses = [i[0] for i in fitness] # i[0] = fitness

            self.avg_per_gen[self.gen] = mean(fitnesses)
            self.median_per_gen[self.gen] = median(fitnesses)
    
    def __repr__(self) -> str:
        from pprint import pformat
        return f"Stats:\nCurrent Generation: {self.gen}\nBest per Generation:\n{pformat(self.best_per_gen)}"
    
    @property
    def total_best(self) -> Tuple[int, float, Individual]:
        best_score = float("-inf")
        best_individual = 0
        best_gen = 0
        for generation, (score, individual) in self.best_per_gen.items():
            if score > best_score:
                best_score = score
                best_individual = individual
                best_gen = generation
        return best_gen, best_score, best_individual
    
    @property
    def best_individual(self) -> Individual:
        return self.total_best[2]
    
    def stats_fromgen(self, start_gen: int, stop_gen: int) -> str:
        from pprint import pformat
        bests = {}
        avg_per_gen = {}
        median_per_gen = {}
        for i in range(start_gen, stop_gen+1):
            bests[i] = self.best_per_gen[i]
            avg_per_gen[i] = self.avg_per_gen[i]
            median_per_gen[i] = self.median_per_gen[i]
        return f"Stats:\nCurrent Generation: {self.gen}\nBest per Generation:\n{pformat(bests)}\nAverage per Generation:\n{pformat(avg_per_gen)}\nMedian per Generation:\n{pformat(median_per_gen)}"
    
    def full_stats(self) -> str:
        from pprint import pformat
        return f"Stats:\nCurrent Generation: {self.gen}\nBest per Generation:\n{pformat(self.best_per_gen)}\nAverage per Generation:\n{pformat(self.avg_per_gen)}\nMedian per Generation:\n{pformat(self.median_per_gen)}"
    
    def empty(self):
        self.avg_per_gen.clear()
        self.best_per_gen.clear()
        self.median_per_gen.clear()
