import numpy as np
from spacepy import pycdf
import matplotlib.pyplot as plt
import datetime

# cdf1 = pycdf.CDF('sep2017/PSD/PSD_rbspb_rept-sci_2017248.cdf').copy()
cdf1 = pycdf.CDF('sep2017/PSD/PSD_rbspb_mageis_2017249.cdf').copy()
# print(cdf1) # 249

# for key in cdf1.keys():
#     print(key, cdf1[key].shape)
#     print(cdf1[key][1])
print(cdf1["OrbTimes"])
nOrbits = len(cdf1["OrbTimes"]) + 1
# print(cdf1["OrbStates"])
# print(cdf1["Epoch"][0], cdf1["Lstar"][0])
# print(cdf1["Epoch"][100], cdf1["Lstar"][100])
# print(cdf1["Epoch"][200], cdf1["Lstar"][200])
# print(cdf1["Epoch"][300], cdf1["Lstar"][300])
# 5/0

def find_orbit(t):
    for i, orb in enumerate(cdf1["OrbTimes"]):
        if t < orb:
            return i
    return nOrbits - 1

def points_from_cdf(cdf, mu_, I_, tolerance):
    shape = cdf["PSD"].shape
    nEpochs, nLstars, nPoints = shape
    # print(nEpochs, nLstars, nPoints)
    points = []
    # mu_ = 700
    # mu_ = 1300
    # I_ = 0.11
    # I_ = 0.15
    for i in range(nEpochs):
        # print(i)
        epoch = cdf["Epoch"][i]
        for j in range(nLstars):
            for k in range(nPoints):
                Lstar = cdf["Lstar"][i, j]
                I = cdf["I"][i, j]
                mu = cdf["mu"][i, j, k]
                PSD = cdf["PSD"][i, j, k]
                if (1-tolerance<=mu/mu_<=1+tolerance):
                    if (1-tolerance<=I/I_<=1+tolerance):
                        if PSD > 0 and Lstar > 0:
                            points.append((epoch, Lstar, I, mu, PSD))
    return points

# points = points_from_cdf(cdf1, 1300, 0.15, 0.16)
points = points_from_cdf(cdf1, 700, 0.11, 0.16)
t0 = min([point[0] for point in points])
tf = max([point[0] for point in points])
print(t0, tf)
date = datetime.date(tf.year, tf.month, tf.day)
hour_points = [[] for i in range(nOrbits)]
for point in points:
    hour_points[find_orbit(point[0])].append(point)
# Lstar = [point[1] for point in points]
# PSD = [point[-1] for point in points]
# plt.plot(Lstar, PSD, "ro")
# plt.show()
# print(min([point[0] for point in points]), max([point[0] for point in points]))
# print(min([point[1] for point in points]), max([point[1] for point in points]))
# print(min([point[-1] for point in points]), max([point[-1] for point in points]))
# print(len(points))

right_boundary = []

for i in range(nOrbits):
    hp = hour_points[i]
    Lstar = [point[1] for point in hp]
    PSD = [point[-1] for point in hp]
    # plt.plot(Lstar, PSD, "ro")
    # plt.show()
    L_range = (3.1, 5.0)
    Li = np.linspace(L_range[0], L_range[1], 81)
    F_bars = []
    for L in Li:
        ps = list(filter(lambda p: L - 0.1 <= p[1] <= L + 0.1, hp))
        ts = [p[0] for p in ps]
        ps = [p[-1] for p in ps]
        F_bars.append(np.mean(ps))
    plt.plot(Li, F_bars, "ro")
    plt.show()


print(Li)
print(F_bars)