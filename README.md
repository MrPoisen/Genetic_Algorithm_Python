# Genetic Algorithm

This is a pure Python implementation for a genetic algorithm. It has precreated classes and functions for using this algorithm.

The genetic module is compossed of a Individual and GeneticAlgorithm class. 
````Python
from genetic_algorithm.genetic import GeneticAlgorithm, Individual
from genetic_algorithm.chooser import DefaultChooser
from genetic_algorithm.rules import single_crossover, SimpleMutator

def get_pop() -> list[Individual]:
    ...

def fitness(indivdual: Individual) -> float: # or int
    ...

ga = GeneticAlgorithm(get_pop(), DefaultChooser(), single_crossover, SimpleMutator(), tracker=..., n_keep=..., n_mutate_keeped=...)
ga.train(iterations=10, fitness_func=fitness)
# or use ga.evololution(fitness, do_sort) where fitness is a list of tuples of a number representing a score and an Individual
# if the list isn't sorted already, pass True for do_sort 
````

When creating your population, you should give the Individuals a list of values, a ``keras.Model``, a ``keras.Sequential`` or a ``NN`` object, as well as a id.  
The ``NN`` object can be created with the simple_nn modul. You will need tensorflow for for ``keras.Model`` and ``keras.Sequential``.