from spoc_delivery_scheduling_evaluate_code import trappist_schedule
import random
import numpy as np
import tqdm
import bisect

N_STATIONS = 12
N_ASTEROIDS = 340

TIME_START = 0
TIME_END = 80

DAY_INTERVAL = 1
ITERATION_TIMES = 100

POPULATION_SIZE = 100

MUTATION_RATE = 0.2

asteroids = {}


class Individual:
    def __init__(self, active_window_index, start_time, calculate_assignment=True) -> None:
        self.active_window_index: np.ndarray = np.copy(active_window_index)
        self.start_time: np.ndarray = np.copy(start_time)
        if calculate_assignment:
            windows = myEA.fill_window(start_time)
            self.assignment_pair = myEA.random_asteroids_assignment(asteroids, windows, active_window_index)
        pass

    def __repr__(self):
        return f"{self.start_time}; {self.active_window_index}"


def crossover(p1: Individual, p2: Individual) -> Individual:
    window_index1, window_index2 = p1.active_window_index, p2.active_window_index
    start_time1, start_time2 = p1.start_time, p2.start_time

    def crossover_simple_recombination(parent1, parent2) -> np.ndarray:
        edgemap: dict[int, set[int]] = {}
        plen = len(parent1)
        for i in range(plen):
            edgemap[i] = set()

        for i in range(plen):
            curr_node = parent1[i]
            edgemap[curr_node].add(parent1[(i - 1) % plen])
            edgemap[curr_node].add(parent1[(i + 1) % plen])

        for i in range(plen):
            curr_node = parent2[i]
            edgemap[curr_node].add(parent2[(i - 1) % plen])
            edgemap[curr_node].add(parent2[(i + 1) % plen])

        for i in range(plen):
            edgemap[i] = list(edgemap[i])

        ret = np.zeros(plen, dtype=np.int32)

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
            if min_decision == plen + 1:
                # no entry available, randomly pick one
                for j in range(plen):
                    if j not in visited:
                        min_entries = plen + 1
                        min_decision = j
                        break
            curr_decision = min_decision
            ret[i + 1] = curr_decision
            visited.append(curr_decision)

        return ret

    window_index_child = crossover_simple_recombination(window_index1, window_index2)

    return Individual(window_index_child, start_time1, calculate_assignment=False)


def selection(populations: list[Individual], fitness: np.ndarray[float]) \
        -> list[Individual, Individual]:
    def roulette(populations: list[Individual], f: np.ndarray[float]) \
            -> list[Individual, Individual]:
        # avoid 0
        f = f - 1
        prob = f / f.sum()
        parents = random.choices(populations, prob, k=2)
        return parents

    return roulette(populations, fitness)


def mutate(pop: Individual) -> Individual:
    def mutateActiveWindowsIndex(x: np.ndarray):
        def swap(xx: np.ndarray):
            ret = xx
            i = random.choices(range(len(ret)), k=2)
            ret[i[0]], ret[i[1]] = ret[i[1]], ret[i[0]]
            return ret

        gene = x
        if random.random() < MUTATION_RATE:
            gene = swap(gene)
        # print(f"\nOrigin: {x}\nFORNOW: {ret}")
        return gene

    new_active_window_index = mutateActiveWindowsIndex(pop.active_window_index)
    new_start_time = np.copy(pop.start_time)
    return Individual(new_active_window_index, new_start_time)


# TODO: Design and implement your EA

class Opportunity:
    def __init__(self, reachTime, massA, massB, massC):
        self.reachTime = reachTime
        self.massA = massA
        self.massB = massB
        self.massC = massC

    def __str__(self):
        return f"\t\tReach Time: {self.reachTime}\n"

    def __repr__(self):
        return f"\t\tReach Time: {self.reachTime}\n"


class Station:
    def __init__(self):
        self.opportunities: dict[int, Opportunity] = {}

    def __str__(self):
        return f"\tStation: {self.opportunities}\n"

    def __repr__(self):
        return f"\tStation: {self.opportunities}\n"


class Asteroids:
    def __init__(self):
        self.stations: dict[int, Station] = {}

    def __str__(self):
        return f"Asteroid: {self.stations}\n"

    def __repr__(self):
        return f"Asteroid: {self.stations}\n"


def find_index_in_window(reach_time: float, window: list[float]) -> int:
    index = bisect.bisect_left(window, reach_time)

    return index // 2 if index % 2 == 1 else -1


class myEA():
    def __init__(self) -> None:
        pass

    @staticmethod
    def uniform_window() -> list:
        """
        @return: A uniformed window for testing
        """
        total_time = TIME_END - TIME_START - (N_STATIONS - 1) * DAY_INTERVAL
        time_each = total_time / N_STATIONS  # time of each station

        uniform_window = []
        for i in range(N_STATIONS):
            uniform_window.append(TIME_START + i * time_each + DAY_INTERVAL * i)  # start time
            uniform_window.append(TIME_START + (i + 1) * time_each + DAY_INTERVAL * i)  # end time
        uniform_window[0] = TIME_START
        uniform_window[-1] = TIME_END

        return uniform_window

    @staticmethod
    def fill_window(start_window: np.ndarray) -> np.ndarray:
        """
        @return window array: [start, end, start, end, ...]
        @param start_window: the start time of current window
        """

        window = np.zeros(2 * N_STATIONS)

        for i in range(N_STATIONS - 1):
            window[2 * i] = start_window[i]
            window[2 * i + 1] = max(start_window[i + 1] - DAY_INTERVAL * 1.05,
                                    start_window[i])  # Make sure no collision happened
        window[-2] = start_window[-1]
        window[-1] = TIME_END

        return window

    @staticmethod
    def window_encoding(window: list, index: np.ndarray) -> np.ndarray:
        ret = np.zeros(2 * N_STATIONS)
        for i in range(N_STATIONS):
            ret[index[i] * 2] = window[2 * i]
            ret[index[i] * 2 + 1] = window[2 * i + 1]
        return ret

    @staticmethod
    def encode(window_encoded: np.ndarray, asteroid_encoded: np.ndarray):
        solution = np.zeros((2 * N_STATIONS + N_ASTEROIDS * 3))

        solution[:2 * N_STATIONS] = window_encoded

        solution[2 * N_STATIONS:] = asteroid_encoded.flatten()

        return solution

    @staticmethod
    def decode_asteroids(db: dict[int, dict]) -> dict[int, Asteroids]:
        asteroids: dict[int, Asteroids] = {}
        for i in range(1, N_ASTEROIDS + 1):
            curr_asteroids: dict = db[i]
            asteroids[i] = Asteroids()
            for j in range(1, N_STATIONS + 1):
                asteroids[i].stations[j] = Station()
                curr_oppo = curr_asteroids[j]
                for k in range(len(curr_oppo)):
                    oppo = curr_oppo[k]
                    asteroids[i].stations[j].opportunities[k + 1] = Opportunity(oppo[0], oppo[1], oppo[2], oppo[3])
        return asteroids

    @staticmethod
    def get_available_assignments(asteroids: dict[int, Asteroids], window: list[float],
                                  window_index: np.ndarray[int]) -> dict[int, list[list[int, int]]]:
        """
        @return {asteroid_id: [[station1, o1], [station1, o2], [station2, o4]]}
        @param asteroids: {asteroid_id: Asteroid}
        @param window: activate window from 0 to 80
        @param window_index: map from window list to station index
        """
        availableAssignments: dict[int, list[list[int, int]]] = {}

        window_index_reverse: dict = {window_index[i]: i for i in range(len(window_index))}

        for i in asteroids:
            availableAssignments[i] = []
            asteroid = asteroids[i]
            for j in asteroid.stations:
                station = asteroid.stations[j]
                win_idx = window_index_reverse[j - 1]
                start, end = window[win_idx * 2], window[win_idx * 2 + 1]
                for k in station.opportunities:
                    oppo = station.opportunities[k]

                    if start <= oppo.reachTime <= end:
                        availableAssignments[i].append([j, k])
                        pass
                    # index = find_index_in_window(reach_time, window)
                    # if index == -1:  # if no suitable station found...
                    #     continue
                    # # 这里必须保证index是从1开始
                    # if window_index[index] + 1 != j:  # current suitable station not suitable for opportunity station
                    #     continue
                    # availableAssignments[i].append([j, k])
        return availableAssignments

    @staticmethod
    def random_asteroids_assignment(asteroids: dict[int, Asteroids], window: list[float],
                                    window_index: np.ndarray[int]) -> \
            list[list[int, int, int]]:
        """
        @return [[asteroid id, station id, opportunity id], [asteroid id, station id, opportunity id] , ... ]
        """

        available_assignments = myEA.get_available_assignments(asteroids, window, window_index)

        assignments = []
        count = 0
        for i in available_assignments:  # i-th asteroids
            assignment: list[list[int, int]] = available_assignments[i]
            if len(assignment) == 0:
                assignments.append([i, 0, 0])
                count = count + 1
            else:
                station, oppo = random.choice(assignment)
                # assignments.append([i, 0, 0])
                assignments.append([i, station, oppo])
        # print(f"Count: {count}")
        return assignments

    @staticmethod
    def fill_min_asteroids_assignment(asteroids: dict[int, Asteroids], window: list[float],
                                      window_index: np.ndarray[int]) -> \
            list[list[int, int, int]]:
        """
        @return [[asteroid id, station id, opportunity id], [asteroid id, station id, opportunity id] , ... ]
        """

        available_assignments = myEA.get_available_assignments(asteroids, window, window_index)

        assignments = []

        station_mass = np.zeros((N_STATIONS + 1, 3))

        station_mass[0] = [1e7, 1e7, 1e7]  # ignore index 0

        count = 0

        for i in available_assignments:  # i-th asteroids
            assignment: list[list[int, int]] = available_assignments[i]  # [[station, oppo], [station, oppo], ... ]
            if len(assignment) == 0:
                assignments.append([i, 0, 0])
                count = count + 1
            elif len(assignment) == 1:
                station, oppo = random.choice(assignment)
                assignments.append([i, station, oppo])
                ooo = asteroids[i].stations[station].opportunities[oppo]
                station_mass[station] += [ooo.massA, ooo.massB, ooo.massC]
            else:
                # find the min value and index of station_mass
                # then if a new oppo added, then the result should be the best

                min_mass = np.min(station_mass)
                min_mass_station = assignment[0][0]
                min_mass_oppo = assignment[0][1]
                for station, oppo in assignment:
                    curr_mass = np.copy(station_mass)

                    ooo = asteroids[i].stations[station].opportunities[oppo]
                    curr_mass[station] += [ooo.massA, ooo.massB, ooo.massC]

                    curr_min_mass = np.min(curr_mass)
                    if curr_min_mass < min_mass:
                        min_mass_station = station_mass
                        min_mass = curr_min_mass

                assignments.append([i, min_mass_station, min_mass_oppo])
                ooo = asteroids[i].stations[min_mass_station].opportunities[min_mass_oppo]
                station_mass[min_mass_station] += [ooo.massA, ooo.massB, ooo.massC]

        return assignments

    @staticmethod
    def mutateActiveWindowsIndex(x: np.ndarray):
        def swap(xx: np.ndarray):
            ret = np.copy(xx)
            i = random.choices(range(len(ret)), k=2)
            ret[i[0]], ret[i[1]] = ret[i[1]], ret[i[0]]
            return ret

        ret = swap(x)
        # print(f"\nOrigin: {x}\nFORNOW: {ret}")
        return ret

    @staticmethod
    def init_population() -> list[Individual]:

        populations = []

        for i in range(POPULATION_SIZE):
            active_windows_index = np.arange(N_STATIONS)
            np.random.shuffle(active_windows_index)

            start_time = np.random.rand(N_STATIONS) * (TIME_END - TIME_START)
            start_time.sort()

            populations.append(Individual(active_windows_index, start_time))

        return populations

    @staticmethod
    def calcSingleFitness(ts, active_windows_index, windows, assignment_pair):
        active_windows = myEA.window_encoding(windows, active_windows_index)
        solution = myEA.encode(active_windows, np.array(assignment_pair))
        return ts.fitness(solution)

    @staticmethod
    def calcFitness(ts: trappist_schedule, populations: list[Individual]) -> np.ndarray[float]:

        fitnesses = np.zeros(POPULATION_SIZE)

        for i in range(POPULATION_SIZE):
            population = populations[i]
            active_windows_index, start_time = population.active_window_index, population.start_time

            windows = myEA.fill_window(start_time)

            # assignment_pair = myEA.random_asteroids_assignment(asteroids, windows, active_windows_index)
            assignment_pair = population.assignment_pair

            fitness, _, _, _, _ = myEA.calcSingleFitness(ts, active_windows_index, windows, assignment_pair)
            fitnesses[i] = fitness
        return fitnesses

    @staticmethod
    def calcSingleFitnessDebug(ts, active_windows_index, start_time, assignment_pair):
        windows = myEA.fill_window(start_time)
        active_windows = myEA.window_encoding(windows, active_windows_index)
        solution = myEA.encode(active_windows, np.array(assignment_pair))
        return ts.fitness(solution)

    @staticmethod
    def calcSingleFitnessDebug2(ts, individual: Individual):
        windows = myEA.fill_window(individual.start_time)
        active_windows = myEA.window_encoding(windows, individual.active_window_index)
        assignment_pair = individual.assignment_pair
        solution = myEA.encode(active_windows, np.array(assignment_pair))
        return ts.fitness(solution)

    @staticmethod
    def main2():
        """
        @descrption: This function is the invocation interface of your EA for testEA.py.
                     Thus you must remain and complete it.
        @return your_decision_vector: the decision vector found by your EA, 1044 dimensions
        """
        # random.seed(5)
        ts: trappist_schedule = trappist_schedule()

        db = ts.db

        # init populations

        global asteroids
        asteroids = myEA.decode_asteroids(db)

        populations: list[Individual] = myEA.init_population()

        fitness = myEA.calcFitness(ts, populations)

        best_fitness_individual = populations[np.argmin(fitness)]
        best_fitness = fitness[np.argmin(fitness)]

        for it in tqdm.tqdm(range(ITERATION_TIMES)):
            # solve window problem
            offspring = []
            for i in range(POPULATION_SIZE):
                # Selection
                parent1, parent2 = selection(populations, fitness)
                # Crossover
                child = crossover(parent1, parent2)
                # Mutation
                child = mutate(child)
                # Local Search, skip for now
                offspring.append(child)
            # TODO: solve asteroids problem

            fitness = myEA.calcFitness(ts, offspring)

            minfit = np.min(fitness)

            if minfit < best_fitness:
                best_fitness = minfit
                best_fitness_individual = offspring[np.argmin(fitness)]

                print(f"Best fitness: {best_fitness}")

                windows = myEA.fill_window(best_fitness_individual.start_time)
                active_windows = myEA.window_encoding(windows, best_fitness_individual.active_window_index)
                assignment_pair = best_fitness_individual.assignment_pair

                solution = myEA.encode(active_windows, np.array(assignment_pair))
                print(ts.fitness(solution))

            populations = offspring
        windows = myEA.fill_window(best_fitness_individual.start_time)
        active_windows = myEA.window_encoding(windows, best_fitness_individual.active_window_index)
        assignment_pair = best_fitness_individual.assignment_pair

        solution = myEA.encode(active_windows, np.array(assignment_pair))

        print(f"Best fitness: {best_fitness}")
        return solution

    @staticmethod
    def main():
        """
        @descrption: This function is the invocation interface of your EA for testEA.py.
                     Thus you must remain and complete it.
        @return your_decision_vector: the decision vector found by your EA, 1044 dimensions
        """

        ts: trappist_schedule = trappist_schedule()

        solution = []

        active_windows_index = np.arange(N_STATIONS)

        np.random.shuffle(active_windows_index)

        best_fitness = 1

        best_solution = None

        db = ts.db
        asteroids = myEA.decode_asteroids(db)

        for it in tqdm.tqdm(range(ITERATION_TIMES)):

            active_windows_index = myEA.mutateActiveWindowsIndex(active_windows_index)

            windows = myEA.uniform_window()

            active_windows = myEA.window_encoding(windows, active_windows_index)

            # assignment_pair = myEA.random_asteroids_assignment(asteroids, active_windows, active_windows_index)
            assignment_pair = myEA.random_asteroids_assignment(asteroids, windows, active_windows_index)
            # assignment_pair = myEA.fill_min_asteroids_assignment(asteroids, windows, active_windows_index)

            solution = myEA.encode(active_windows, np.array(assignment_pair))

            fitness, _, _, _, _ = ts.fitness(solution)
            # print(f"fitness: {fitness}")
            if fitness < best_fitness:
                best_fitness = fitness
                best_solution = solution

        return best_solution


if __name__ == "__main__":
    udp = trappist_schedule()
    your_decision_vector = myEA.main2()
    # fitness, wrongly indexed asteroids, assignment violations
    # minimum time gap, violations of constraint among the allocated asteroids
    fitness_values = udp.fitness(your_decision_vector)

    print(fitness_values)

    # print(myEA.calculate_window(np.arange(0, 24)))
    # window = myEA.uniformWindow()
    # print(window)
    # print(myEA.windowEncoding(window, np.arange(N_STATIONS)))
