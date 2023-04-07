from spoc_delivery_scheduling_evaluate_code import trappist_schedule
import random
import numpy as np
import tqdm

N_STATIONS = 12
N_ASTEROIDS = 340

TIME_START = 0
TIME_END = 80

DAY_INTERVAL = 1
ITERATION_TIMES = 100


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
    for i in range(len(window) // 2):
        if window[2 * i] <= reach_time <= window[2 * i + 1]:
            return i
    return -1


class myEA():
    def __init__(self) -> None:
        pass

    @staticmethod
    def calculate_window(window_time: np.ndarray[float]) -> np.ndarray[float]:
        # window_time: [T 1i, T 1f, ..., T 12i, T 12f]
        window_start = window_time[0:2 * N_STATIONS:2]
        window_end = window_time[1:2 * N_STATIONS:2]

        index = np.argsort(window_start)

        window_start[index[0]] = TIME_START  # set first window to 0

        for i in range(0, N_STATIONS - 1):
            window_end[index[i]] = window_start[index[i + 1]] + DAY_INTERVAL

        window_end[index[N_STATIONS - 1]] = TIME_END

        result = []

        for i in range(0, N_STATIONS):
            result.append(window_start[i])
            result.append(window_end[i])

        return np.array(result)

    @staticmethod
    def uniform_window() -> list:
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

        for i in asteroids:
            availableAssignments[i] = []
            asteroid = asteroids[i]
            for j in asteroid.stations:
                station = asteroid.stations[j]

                for k in station.opportunities:
                    oppo = station.opportunities[k]
                    reach_time = oppo.reachTime
                    index = find_index_in_window(reach_time, window)
                    if index == -1:  # if no suitable station found...
                        continue
                    # 这里必须保证index是从1开始
                    if window_index[index] + 1 != j:  # current suitable station not suitable for opportunity station
                        continue
                    availableAssignments[i].append([j, k])
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

        station_mass = np.zeros((N_STATIONS, 3))

        count = 0

        for i in available_assignments:  # i-th asteroids
            assignment: list[list[int, int]] = available_assignments[i]
            if len(assignment) == 0:
                assignments.append([i, 0, 0])
                count = count + 1
            else:
                min_mass_idxes: list[list[int, int, int]] = np.argsort(station_mass)
                fm_station = 0 #fill min station
                fm_oppo = 0 # fill min opportunity
                min_mass_idx = 0
                min_mass: list[float, float, float] = station_mass[min_mass_idxes[min_mass_idx, 0]]
                for station, oppo in assignment:
                    if min_mass > station_mass[min_mass_idxes[min_mass_idx, 0]]:
                        min_mass = station_mass[min_mass_idxes[min_mass_idx, 0]]
                        min_mass_idx = min_mass_idxes
                    pass
                station, oppo = random.choice(assignment)
                # assignments.append([i, 0, 0])
                assignments.append([i, station, oppo])
        # print(f"Count: {count}")
        return assignments

    @staticmethod
    def mutateActiveWindowsIndex(x: np.ndarray):
        def swap(xx: np.ndarray):
            i = random.choices(range(len(xx)), k=2)
            xx[i[0]], xx[i[1]] = xx[i[1]], xx[i[0]]
            return xx

        return swap(x)

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

        best_fitness = 0

        best_solution = None

        for it in tqdm.tqdm(range(ITERATION_TIMES)):

            db = ts.db

            active_windows_index = myEA.mutateActiveWindowsIndex(active_windows_index)

            windows = myEA.uniform_window()

            active_windows = myEA.window_encoding(windows, active_windows_index)

            asteroids = myEA.decode_asteroids(db)

            # assignment_pair = myEA.random_asteroids_assignment(asteroids, active_windows, active_windows_index)
            # assignment_pair = myEA.random_asteroids_assignment(asteroids, windows, active_windows_index)
            assignment_pair = myEA.fill_min_asteroids_assignment(asteroids, windows, active_windows_index)

            solution = myEA.encode(active_windows, np.array(assignment_pair))

            fitness, _, _, _, _ = ts.fitness(solution)
            print(f"fitness: {fitness}")
            if fitness < best_fitness:
                best_fitness = fitness
                best_solution = solution

        return best_solution


if __name__ == "__main__":
    udp = trappist_schedule()
    your_decision_vector = myEA.main()
    fitness_values = udp.fitness(your_decision_vector)

    print(fitness_values)

    # print(myEA.calculate_window(np.arange(0, 24)))
    # window = myEA.uniformWindow()
    # print(window)
    # print(myEA.windowEncoding(window, np.arange(N_STATIONS)))
