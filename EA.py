from spoc_delivery_scheduling_evaluate_code import trappist_schedule
import random
import numpy as np

#TODO: Design and implement your EA
class myEA():
    def __init__(self) -> None:
        pass

    def main():
        '''
        @descrption: This function is the invocation interface of your EA for testEA.py.
                     Thus you must remain and complete it.
        @return your_decision_vector: the decision vector found by your EA, 1044 dimensions
        '''

        ts : trappist_schedule = trappist_schedule()

        n_stations = ts.n_stations
        n_asteriods = ts.n_asteroids
        bounds = ts.get_bounds
        print(bounds)
        gap = ts.station_gap
        db = ts.db

        active_windows = np.zeros((2*n_stations))

        start_time = 0

        for i in range(n_stations):
            active_windows[2*i]=start_time # start time
            active_windows[2*i+1]=start_time + 1 # end time
            start_time = start_time + 1 + gap
        
        assignment_pair = np.zeros((n_asteriods, 3))

        for i in range(n_asteriods):
            index = i+1
            station_id = random.randint(0, n_stations) # 0 means unassigned
            opportunity_id = random.randint(0, 8)
            assignment_pair[i] = [index, station_id, opportunity_id]


        solution = np.zeros((2*n_stations+n_asteriods*3))

        solution[:2*n_stations] = active_windows

        solution[2*n_stations:] = assignment_pair.flatten()
        
        return solution
    
if __name__ == "__main__":
    udp = trappist_schedule()
    your_decision_vector = myEA.main()
    fitness_values = udp.fitness(your_decision_vector)
    print(fitness_values)