import matplotlib.pyplot as plt
import numpy as np
from read_Kp import read_Kp
from kalman import timeToDays
import json

dir_path = "week"
with open(dir_path + '/kalman_input.json', "r") as inputJSON:
    kalman_data = json.load(inputJSON)
L_range = kalman_data["L_range"]
t_range = kalman_data["t_range"]
orbits = kalman_data["orbits"]
orbit_boundary_times = [np.array(orbit, dtype=float) for orbit
                        in kalman_data["orbit_boundary_times"]]
orbit_log_lefts = [np.array(orbit, dtype=float)
                   for orbit in kalman_data["orbit_log_lefts"]]
orbit_log_rights = [np.array(orbit, dtype=float)
                    for orbit in kalman_data["orbit_log_rights"]]
log_initial = np.array(kalman_data["log_initial"], dtype=float)
VAP_times = np.array(kalman_data["VAP_times"], dtype=float)
VAP_logs = np.array(kalman_data["VAP_logs"], dtype=float)
H_all_data = np.array(kalman_data["H"], dtype=float)
model_Li = np.array(kalman_data["model_Li"], dtype=float)


Kp_data = read_Kp(dir_path + "/Kp_data.lst")
Kp_data = {timeToDays(t): Kp for t, Kp in Kp_data.items()}
Kp_times = list(sorted(Kp_data.keys()))
assert(Kp_times != [])
plt.title("Kp data")
plt.plot([t/24+1 for t in Kp_times], Kp_data.values())
plt.xlabel('Time (days)')
plt.ylabel('Kp')
plt.show()
