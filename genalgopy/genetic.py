from statistics import mean
from typing import List, Tuple, Union

try:
    from tensorflow import keras
    TENSORFLOW = True
except ImportError:
    TENSORFLOW = False

import numpy as np

from ._keras import vectorize, from_vector, modelcopy, _is_tensornetwork
from .simple_nn import NN

#import dill as pickle
import pickle

def _get_fitness(pop, fitness_func):
    fitness = []
    for individual in pop:
        fitness.append((fitness_func(individual), individual))
    fitness.sort(key=lambda x: x[0], reverse=True)
    return fitness

class Individual:
    def __init__(self, genoms: Union[list, "keras.Sequential", NN], id_: int) -> None:
        self.genoms = genoms
        self.id = id_

    def __repr__(self) -> str:
        return f"Individual(id={self.id})"

    def __eq__(self, o: object) -> bool:
        return False if not isinstance(o, Individual) else o.id == self.id

    def __hash__(self) -> int:
        return hash(self.id)

    def edit_genom(self, genoms: list):
        """
        Args:
            genoms: replaces current genom
        """
        if _is_tensornetwork(self.genoms):
            from_vector(self.genoms, genoms)
        elif isinstance(self.genoms, NN):
            self.genoms.from_vector(genoms)
        else:
            self.genoms = genoms

    def get_genome(self) -> list:
        """
        Returns:
            list of current genoms
        """
        if _is_tensornetwork(self.genoms):
            return vectorize(self.genoms)
        elif isinstance(self.genoms, NN):
            return self.genoms.vectorize()
        return self.genoms

    def copy(self, id_: int) -> "Individual":
        """
        Args:
            id_: id for the copy
        Returns:
            an Individual with the same genoms as the current one
        """
        if _is_tensornetwork(self.genoms):
            genom = modelcopy(self.genoms)
        else:
            genom = self.genoms.copy()
        return Individual(genom, id_)

    def execute(self, x: list, default=None):
        """
        Args:
            x: input for a keras.Sequential or NN
            default: will be returned if the genoms are a list (not a Neural Network)
        Returns:
            default or the output of a Neural Network
        """
        arr = np.array(x)
        if _is_tensornetwork(self.genoms):
            multi = True
            if len(arr.shape) == 1:
                arr = np.array([arr])
                multi = False
            res = np.array(self.genoms(arr))
            return res[0] if multi is False else res
        elif isinstance(self.genoms, NN):
            return self.genoms(x)
        return default

    def save(self) -> bytes:
        """
        Returns:
            Individual as bytes
        """
        genom_type = "list"
        genom = self.genoms
        if _is_tensornetwork(self.genoms):
            genom_type = "tensorflow"
            genom = (self.genoms.to_json(), self.genoms.get_weights())
        elif isinstance(self.genoms, NN):
            genom_type = "simple_nn"
            genom = self.genoms.save()
        return pickle.dumps((genom_type, genom, self.id))

    @staticmethod
    def load(data: bytes) -> "Individual":
        """
        Args:
            data: bytes representation of an Individual
        Returns:
            an Individual based on data
        """
        genom_type, genom, id_ = pickle.loads(data)
        if genom_type == "list":
            return Individual(genom, id_)
        elif genom_type == "tensorflow":
            if TENSORFLOW is False:
                raise Exception("can't load Individual because tensorflow couldn't be imported")
            model = keras.models.model_from_json(genom[0])
            model.set_weights(genom[1])
            return Individual(model, id_)
        elif genom_type == "simple_nn":
            model = NN.load(genom)
            return Individual(model, id_)
        else:
            raise Exception("unknown genom_type")


class GeneticAlgorithm:
    def __init__(self, population: List[Individual], chooser, kidsmaker, mutator, tracker: "Tracker" =None, keep=2, n_mutate_keeped: int = 0) -> None:
        """
        Args:
            population: list of Individuals to evolve
            chooser: object that decides which two individuals to choose for childmaking
            kidsmaker: creates two new Individuals based of two parents
            mutator: mutates Individuals
            tracker: tracks the fitness
            keep: amount of best Individuals that should be kept in the new population
            n_mutate_keeped: amount of least best keeped Individals that should mutate
        """
        self.pop = population
        self.kidsmaker = kidsmaker
        self.mutator = mutator
        self.chooser = chooser

        self._keep = keep
        self.n_mutate_keeped = n_mutate_keeped
        self.id = max(individual.id for individual in self.pop) + 1
        self.tracker = tracker
        #if self.tracker is not None:
            #self.tracker(self.pop)

        if (len(self.pop) - self._keep) % 2 != 0:
            raise ValueError("value for keep must allow an even rest-population")

        if len(self.pop) < 1:
            raise Exception("You must give a population of size > 0")

    def save_population(self) -> bytes:
        """
        returns bytes representing the population
        """
        gen = self.tracker.gen if self.tracker is not None else None
        pop = [i.save() for i in self.pop]
        return pickle.dumps((gen, self.id, pop))

    def load_population(self, data: bytes):
        """
        loads the population from bytes
        """
        gen, id_, pop = pickle.loads(data)
        self.id = id_
        self.pop = [Individual.load(i) for i in pop]
        if self.tracker is not None:
            self.tracker.empty()
            self.tracker.gen = gen
            #self.tracker(self.pop, True)

    def train(self, iterations: int, fitness_func):
        """
        Args:
            iterations: how often the population should evolve
            fitness_func: function that evaluates fitness of an Individual
        """
        for i in range(iterations):
            fitness = _get_fitness(self.pop, fitness_func)
            self.evolution(fitness)
        if self.tracker is not None:
            self.tracker(_get_fitness(self.pop, fitness_func))

    def train_till_single_best(self, fitness_cap: Union[float, int], fitness_func, max_iterations: Union[float, int] = float("inf")) -> int:
        """
        Args:
            fitness_cap: value the best Individual should achieve
            fitness_func: function that evaluates fitness of an Individual
            max_iterations: maximum amount of evolutions of the population before exiting the function
        Returns:
            amount of evolutions that were needed to achieve the fitness_cap 
        """
        i = 0
        while i < max_iterations:
            fitness = _get_fitness(self.pop, fitness_func)
            if fitness[0][0] >= fitness_cap:
                break
            self.evolution(fitness)
            i += 1

        if self.tracker is not None:
            self.tracker(_get_fitness(self.pop, fitness_func))
        return i

    def train_till_average_best(self, fitness_cap: Union[float, int], fitness_func, max_iterations: Union[float, int] = float("inf")) -> int:
        """
        Args:
            fitness_cap: value the population should achieve on average
            fitness_func: function that evaluates fitness of an Individual
            max_iterations: maximum amount of evolutions of the population before exiting the function
        Returns:
            amount of evolutions that were needed to achieve the fitness_cap 
        """
        i = 0
        while i < max_iterations:
            fitness = _get_fitness(self.pop, fitness_func)
            fitnesses = [i[0] for i in fitness] # i[0] = fitness
            if mean(fitnesses) >= fitness_cap:
                break
            self.evolution(fitness)
            i += 1

        if self.tracker is not None:
            self.tracker(_get_fitness(self.pop, fitness_func))
        return i

    def evolution(self, fitness: List[Tuple[float, Individual]], do_sort=False):
        """
        Args:
            fitness: list of Individuals and there fitness value
            do_sort: if the fitness list should be sorted (if it wasn't already)
        """
        if do_sort:
            fitness.sort(key=lambda x: x[0], reverse=True)
        size = len(self.pop)
        weights = np.empty(len(fitness))
        individuals = []
        dictionized = {}
        for index, (score, individual) in enumerate(fitness):
            weights[index] = score
            individuals.append(individual)
            dictionized[individual] = score

        new_pow = individuals[:self._keep]

        individuals, weights = self.chooser.prepare(individuals, weights)

        while len(new_pow) < size:
            choosen1, choosen2 = self.chooser(individuals, weights)
            kid1, kid2 = self.kidsmaker((dictionized.get(choosen1), choosen1), (dictionized.get(choosen2), choosen2), self.id)
            self.id += 2
            kid1, kid2 = self.mutator(kid1), self.mutator(kid2)
            new_pow.extend([kid1, kid2])
        if self.n_mutate_keeped > 0:
            for index, indiv in enumerate(new_pow[self._keep-self.n_mutate_keeped:self._keep]):
                new_pow[index+self._keep] = self.mutator(indiv)
        self.pop = new_pow

        if self.tracker is not None:
            self.tracker(fitness)
