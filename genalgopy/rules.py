from typing import Tuple
import random

from .genetic import Individual

def single_crossover(parent1: Tuple[float, Individual], parent2: Tuple[float, Individual], cur_id: int):
    genom1, genom2 = list(parent1[1].get_genome()), list(parent2[1].get_genome())
    position = random.randint(0, len(genom1)-1)

    genom1, genom2 = genom1[0: position] + genom2[position:], genom2[0: position] + genom1[position:]
    kid1, kid2 = parent1[1].copy(cur_id+1), parent2[1].copy(cur_id+2)
    kid1.edit_genom(genom1)
    kid2.edit_genom(genom2)
    return kid1, kid2

def nk_crossover(k: int):
    def crossover(parent1: Tuple[float, Individual], parent2: Tuple[float, Individual], cur_id: int):
        genom1, genom2 = list(parent1[1].get_genome()), list(parent2[1].get_genome())
        positions = [i for i in range(len(genom1))]
        use_positions = random.sample(positions, k)
        use_positions.sort()
        newgenom1, newgenom2 = [], []

        start_idx = 0
        for pos in use_positions:
            newgenom1.extend(genom1[start_idx:pos])
            newgenom2.extend(genom2[start_idx:pos])

            genom1, genom2 = genom2, genom1
            start_idx = pos
        newgenom1.extend(genom1[start_idx:])
        newgenom2.extend(genom2[start_idx:])

        kid1, kid2 = parent1[1].copy(cur_id+1), parent2[1].copy(cur_id+2)
        kid1.edit_genom(newgenom1)
        kid2.edit_genom(newgenom2)

        return kid1, kid2
    return crossover

def equal_uniform_crossover(parent1: Tuple[float, Individual], parent2: Tuple[float, Individual], cur_id: int):
    genom1, genom2 = parent1[1].get_genome(), parent2[1].get_genome()
    for i in range(len(genom1)):
        v_1 = genom1[i]
        v_2 = genom2[i]

        res = v_1 if random.randint(0, 1) == 0 else v_2
        genom1[i] = res
        if res == v_1:
            genom2[i] = v_2
        else:
            genom2[i] = v_1
    
    kid1, kid2 = parent1[1].copy(cur_id+1), parent2[1].copy(cur_id+2)

    kid1.edit_genom(genom1)
    kid2.edit_genom(genom2)
    return kid1, kid2

def simplebias_uniform_crossover(parent1: Tuple[float, Individual], parent2: Tuple[float, Individual], cur_id: int):
    genom1, genom2 = parent1[1].get_genome(), parent2[1].get_genome()
    for i in range(len(genom1)):
        v_1 = genom1[i]
        v_2 = genom2[i]
        res = random.choices((v_1, v_2), (parent1[0], parent2[0]), k=1)[0]
        genom1[i] = res
        if res == v_1:
            genom2[i] = v_2
        else:
            genom2[i] = v_1
    
    kid1, kid2 = parent1[1].copy(cur_id+1), parent2[1].copy(cur_id+2)

    kid1.edit_genom(genom1)
    kid2.edit_genom(genom2)
    return kid1, kid2

def partially_mapped_crossover(parent1: Tuple[float, Individual], parent2: Tuple[float, Individual], cur_id: int): # TODO: might not work
    map_1, map_2 = {}, {}
    genom1, genom2 = parent1[1].get_genome(), parent2[1].get_genome()
    parent_1_genom, parent_2_genom = genom1.copy(), genom2.copy()

    position1 = random.randint(0, len(parent_1_genom)-2)
    position2 = random.randint(position1+1, len(parent_1_genom)-1)

    for i in range(position1, position2):
        genom1[i] = parent_2_genom[i]
        map_1[parent_2_genom[i]] = parent_1_genom[i]

        genom2[i] = parent_1_genom[i]
        map_2[parent_1_genom[i]] = parent_2_genom[i]
    
    for i in range(0, position1):
        while genom1[i] in map_1.keys():
            genom1[i] = map_1[genom1[i]]

        while genom2[i] in map_2.keys():
            genom2[i] = map_2[genom2[i]]

    for i in range(position2, len(parent_1_genom)):
        while genom1[i] in map_1.keys():
            genom1[i] = map_1[genom1[i]]

        while genom2[i] in map_2.keys():
            genom2[i] = map_2[genom2[i]]

    kid1, kid2 = parent1[1].copy(id_=cur_id+1), parent2[1].copy(id_=cur_id+2)
    kid1.edit_genom(genom1)
    kid2.edit_genom(genom2)
    return kid1, kid2

class SimpleMutator:
    def __init__(self, lower: float, upper: float, amount: int = 1, decimal_places: int = 0, replace: bool = True) -> None:
        self.lower = float(lower)
        self.upper = float(upper)
        self.amount = amount
        self.decimal_places = decimal_places
        self.replace = replace

        if not (self.lower * pow(10, self.decimal_places)).is_integer():
            raise Exception("lower value can't be expressed as an int because decimal_places is too low")
        if not (self.upper * pow(10, self.decimal_places)).is_integer():
            raise Exception("upper value can't be expressed as an int because decimal_places is too low")
    
    def __call__(self, individual: Individual):
        genom = list(individual.get_genome())
        valid_positions = [i for i in range(0, len(genom))]
        for i in range(self.amount):
            position = valid_positions.pop(random.randint(0, len(valid_positions)-1))            
            value = random.randint(self.lower * pow(10, self.decimal_places), self.upper * pow(10, self.decimal_places))
            if not self.decimal_places == 0:
                value /= self.decimal_places
            
            if self.replace:
                genom[position] = value
            else:
                genom[position] += value
            
        individual.edit_genom(genom)
        return individual
