import math

import numpy.random

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

ts: trappist_schedule = trappist_schedule()


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


asteroids: dict[int, Asteroids] = {}


class myEA():

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
    def check_window(window: np.ndarray[float]) -> bool:
        activate_window = window.reshape((N_STATIONS, 2))

        result = sorted(activate_window, key=lambda a: a[0])

        for i in range(len(activate_window) - 1):
            start_time, end_time = result[i]

            if not (TIME_START <= start_time <= TIME_END and TIME_START <= end_time <= TIME_END):
                return False

            n_start_time = result[i + 1]

            if n_start_time - end_time < DAY_INTERVAL:
                return False

        return True

    @staticmethod
    def calc_window(population: list[list[int, int]]) -> np.ndarray[float]:
        global asteroids

        station_time: np.ndarray[int, float] = -1 * np.ones((N_STATIONS, 2))  # record max and min time of a station

        for i in range(len(population)):
            asteroid_id = i + 1
            station, oppo = population[i]

            r_time = asteroids[asteroid_id].stations[station].opportunities[oppo].reachTime

            if station_time[station - 1, 0] == -1:
                station_time[station - 1, 0] = r_time
            else:
                if r_time < station_time[station - 1, 0]:
                    station_time[station - 1, 0] = r_time

            if station_time[station - 1, 1] == -1:
                station_time[station - 1, 1] = r_time
            else:
                if r_time > station_time[station - 1, 1]:
                    station_time[station - 1, 1] = r_time

        return station_time.flatten()

    @staticmethod
    def init_population(ts) -> list[list[list[int, int]]]:
        global asteroids
        asteroids = {}

        populations = []

        for p in tqdm.tqdm(range(POPULATION_SIZE)):
            individual = []
            for i in asteroids:
                stations = asteroids[i].stations
                station = random.choice(stations)
                opportunity = random.choice(station.opportunities)
                individual.append([station, opportunity])

            fitness, _, _, _, _ = ts.fitness(myEA.individualEncoding(individual))
            while fitness == 0:
                individual = []
                for i in asteroids:
                    stations = asteroids[i].stations
                    station = random.choice(stations)
                    opportunity = random.choice(station.opportunities)
                    individual.append([station, opportunity])
                fitness, _, _, _, _ = ts.fitness(myEA.individualEncoding(individual))

            populations.append(individual)
        return populations

    @staticmethod
    def encode(window_encoded: np.ndarray, asteroid_encoded: np.ndarray):
        solution = np.zeros((2 * N_STATIONS + N_ASTEROIDS * 3))

        solution[:2 * N_STATIONS] = window_encoded

        solution[2 * N_STATIONS:] = asteroid_encoded.flatten()

        return solution

    @staticmethod
    def calcSingleFitness(ts, window, assignment_pair):
        solution = myEA.encode(window, np.array(assignment_pair))
        return ts.fitness(solution)

    @staticmethod
    def individualEncoding(individual: list[list[int, int]]):
        window = myEA.calc_window(individual)

        encoder = np.zeros((N_ASTEROIDS * 3))
        for i in range(len(individual)):
            aster = i + 1
            encoder[3*i] = aster
            encoder[3*i+1] = individual[i][0]
            encoder[3*i+2] = individual[i][1]

        solution = np.zeros((2 * N_STATIONS + N_ASTEROIDS * 3))

        solution[:2 * N_STATIONS] = window

        solution[2 * N_STATIONS:] = encoder

        return solution

    @staticmethod
    def calcFitness(ts: trappist_schedule, populations: list) -> np.ndarray[float]:

        fitness = np.zeros(POPULATION_SIZE)

        for i in range(POPULATION_SIZE):
            individual = populations[i]
            encoding = myEA.individualEncoding(individual)

            f, _, _, _, _ = ts.fitness(encoding)
            fitness[i] = f
        return fitness

    @staticmethod
    def main():
        """
        @descrption: This function is the invocation interface of your EA for testEA.py.
                     Thus you must remain and complete it.
        @return your_decision_vector: the decision vector found by your EA, 1044 dimensions
        """
        # random.seed(5)

        db = ts.db

        # init populations

        global asteroids
        asteroids = myEA.decode_asteroids(db)

        populations = myEA.init_population(ts)

        fitness = myEA.calcFitness(ts, populations)

        best_fitness_individual = populations[np.argmin(fitness)]
        best_fitness = fitness[np.argmin(fitness)]

        # for it in tqdm.tqdm(range(ITERATION_TIMES)):
        #     # solve window problem
        #     offspring = []
        #     for i in range(POPULATION_SIZE):
        #         # Selection
        #         parent1, parent2 = selection(populations, fitness)
        #         # Crossover
        #         child = crossover(parent1, parent2)
        #         # Mutation
        #         child = mutate(child)
        #         offspring.append(child)
        #     # TODO: solve asteroids problem
        #
        #     # Local Search, skip for now
        #
        #     fitness = myEA.calcFitness(ts, offspring)
        #     offspring = local_search(offspring, fitness)
        #     fitness = myEA.calcFitness(ts, offspring)
        #
        #     minfit = np.min(fitness)
        #
        #     if minfit < best_fitness:
        #         best_fitness = minfit
        #         best_fitness_individual = offspring[np.argmin(fitness)]
        #
        #         print(f"Best fitness: {best_fitness}")
        #
        #         windows = myEA.fill_window(best_fitness_individual.start_time)
        #         active_windows = myEA.window_encoding(windows, best_fitness_individual.active_window_index)
        #         assignment_pair = best_fitness_individual.assignment_pair
        #
        #         solution = myEA.encode(active_windows, np.array(assignment_pair))
        #         print(ts.fitness(solution))
        #
        #     populations = offspring

        solution = myEA.individualEncoding(best_fitness_individual)

        print(f"Best fitness: {best_fitness}")
        return solution


if __name__ == "__main__":
    udp = trappist_schedule()
    your_decision_vector = myEA.main()
    # fitness, wrongly indexed asteroids, assignment violations
    # minimum time gap, violations of constraint among the allocated asteroids
    fitness_values = udp.fitness(your_decision_vector)

    print(fitness_values)

    # print(myEA.calculate_window(np.arange(0, 24)))
    # window = myEA.uniformWindow()
    # print(window)
    # print(myEA.windowEncoding(window, np.arange(N_STATIONS)))
