import numpy as np
import random

def crossover(parent1, parent2):
    edgemap: dict[int, set[int]] = {}
    plen = len(parent1)
    for i in range(plen):
        edgemap[i] = set()
        
    for i in range(len(parent1)):
        curr_node = parent1[i]
        edgemap[curr_node].add(parent1[(i-1)%plen])
        edgemap[curr_node].add(parent1[(i+1)%plen])
    
    for i in range(len(parent2)):
        curr_node = parent2[i]
        edgemap[curr_node].add(parent2[(i-1)%plen])
        edgemap[curr_node].add(parent2[(i+1)%plen])
        
    for i in range(plen):
        edgemap[i] = list(edgemap[i])
        
    ret = np.zeros(plen)
    
    visited = []
    
    curr_decision = random.randint(0, plen - 1)
    ret[0] = curr_decision
    
    visited.append(curr_decision)
    
    for i in range(plen - 1):
        min_decision = plen + 1
        min_entries = plen + 1
        for decision in edgemap[curr_decision]:
            entry = len(edgemap[decision])
            if decision not in visited and entry < min_entries:
                min_entries = entry
                min_decision = decision
        curr_decision = min_decision
        ret[i+1] = curr_decision
        visited.append(curr_decision)
    
    return ret


if __name__ == "__main__":
    # p1 = np.arange(7)
    # p2 = np.arange(7)
    p1 = np.array([0,1,2,3,4,5,6,7])
    p2 = np.array([0,1,2,7,3,6,4,5])
    # np.random.shuffle(p2)
    print(p1)
    print(p2)
    print(crossover(p1, p2))