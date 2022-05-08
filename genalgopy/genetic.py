from statistics import mean
from typing import List, Tuple, Union

from tensorflow import keras
import numpy as np

from ._keras import vectorize, from_vector, modelcopy
from .simple_nn import NN

#import dill as pickle
import pickle

def _is_tensornetwork(obj):
    return isinstance(obj, (keras.Sequential, keras.Model)) or issubclass(type(obj), keras.Model)

def _get_fitness(pop, fitness_func):
    fitness = []
    for individual in pop:
        fitness.append((fitness_func(individual), individual))
    fitness.sort(key=lambda x: x[0], reverse=True)
    return fitness

class Individual:
    def __init__(self, genoms: Union[list, keras.Sequential], id_: int) -> None:
        self.genoms = genoms
        self.id = id_

    def __repr__(self) -> str:
        return f"Individual(id={self.id})"

    def __eq__(self, o: object) -> bool:
        return False if not isinstance(o, Individual) else o.id == self.id

    def __hash__(self) -> int:
        return hash(self.id)

    def edit_genom(self, genom):
        if _is_tensornetwork(self.genoms):
            from_vector(self.genoms, genom)
        else:
            self.genoms = genom

    def get_genome(self):
        if _is_tensornetwork(self.genoms):
            return vectorize(self.genoms)
        return self.genoms

    def copy(self, id_):
        if _is_tensornetwork(self.genoms):
            genom = modelcopy(self.genoms)
        else:
            genom = self.genoms.copy()
        return Individual(genom, id_)

    def execute(self, x, default=None):
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
        genom_type, genom, id_ = pickle.loads(data)
        if genom_type == "list":
            return Individual(genom, id_)
        elif genom_type == "tensorflow":
            model = keras.models.model_from_json(genom[0])
            model.set_weights(genom[1])
            return Individual(model, id_)
        elif genom_type == "simple_nn":
            model = NN.load(genom)
            return Individual(model, id_)
        else:
            raise Exception("unknown genom_type")


class GeneticAlgorithm:
    def __init__(self, population: List[Individual], chooser, kidsmaker, mutator, tracker: "Tracker" =None, keep=2, n_mutated_keeped: int = 0) -> None:
        self.pop = population
        self.kidsmaker = kidsmaker
        self.mutator = mutator
        self.chooser = chooser

        self._keep = keep
        self.n_mutated_keeped = n_mutated_keeped
        self.id = max(individual.id for individual in self.pop) + 1
        self.tracker = tracker
        #if self.tracker is not None:
            #self.tracker(self.pop)

        if (len(self.pop) - self._keep) % 2 != 0:
            raise ValueError("value for keep must allow an even rest-population")

        if len(self.pop) < 1:
            raise Exception("You must give a population of size > 0")

    def save_population(self) -> bytes:
        gen = self.tracker.gen if self.tracker is not None else None
        pop = [i.save() for i in self.pop]
        return pickle.dumps((gen, self.id, pop))

    def load_population(self, data: bytes):
        gen, id_, pop = pickle.loads(data)
        self.id = id_
        self.pop = [Individual.load(i) for i in pop]
        if self.tracker is not None:
            self.tracker.empty()
            self.tracker.gen = gen
            #self.tracker(self.pop, True)

    def train(self, iterations, fitness_func):
        for i in range(iterations):
            fitness = _get_fitness(self.pop, fitness_func)
            self.evolution(fitness)
        if self.tracker is not None:
            self.tracker(_get_fitness(self.pop, fitness_func))

    def train_till_single_best(self, fitness_cap: Union[float, int], fitness_func, max_iterations: Union[float, int] = float("inf")) -> int:
        i = 0
        while True:
            fitness = _get_fitness(self.pop, fitness_func)
            if fitness[0][0] >= fitness_cap:
                break
            self.evolution(fitness)
            i += 1
            if i >= max_iterations:
                break

        if self.tracker is not None:
            self.tracker(_get_fitness(self.pop, fitness_func))
        return i

    def train_till_average_best(self, fitness_cap: Union[float, int], fitness_func, max_iterations: Union[float, int] = float("inf")) -> int:
        i = 0
        while True:
            fitness = _get_fitness(self.pop, fitness_func)
            fitnesses = [i[0] for i in fitness] # i[0] = fitness
            if mean(fitnesses) >= fitness_cap:
                break
            self.evolution(fitness)
            i += 1
            if i >= max_iterations:
                break
        if self.tracker is not None:
            self.tracker(_get_fitness(self.pop, fitness_func))
        return i

    def evolution(self, fitness: List[Tuple[float, Individual]], do_sort=False):
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
        if self.n_mutated_keeped > 0:
            for index, indiv in enumerate(new_pow[self._keep-self.n_mutated_keeped:self._keep]):
                new_pow[index+self._keep] = self.mutator(indiv)
        self.pop = new_pow

        if self.tracker is not None:
            self.tracker(fitness)
