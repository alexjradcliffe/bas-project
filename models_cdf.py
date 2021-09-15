import numpy as np
from spacepy import pycdf
import matplotlib.pyplot as plt
import os
import datetime
import math
import json

MINTIME = datetime.datetime(datetime.MINYEAR, 1, 1)

def timeToDays(t):
    """
    Takes a time (t) in the form of datetime.datetime and returns the (float)
    number of days + 1 since midnight on 1st January of that year (jan1,
    provided as a datetime.datetime.
    """
    delta = t-MINTIME
    return delta.total_seconds()/86400

def find_orbit(OrbTimes, t):
    """
    Takes a list OrbTimes of times at which orbits ended, and a time,
    and returns which orbit that time was during. OrbTimes must already be
    sorted before this function is called.
    The way that the numbering works is that if t is before any of the
    OrbTimes, it is counted as being in orbit 0, if it is later than all the
    OrbTimes, it is counted as being in orbit len(OrbTimes).
    Otherwise if OrbTimes[i-1] <= t < OrbTimes[i], it is counted as being in
    Orbit i.
    """
    for i, orb in enumerate(OrbTimes):
        if t < orb:
            return i
    return len(OrbTimes)

def points_from_cdf(cdf, mu_, I_, tolerance):
    """
    Takes cdf, a cdf-file with data from https://rbspgway.jhuapl.edu/psd about
    the phase space density as a function of L-star, mu and I; values mu_
    and I_, which are the values of mu and I at which we are considering the
    phase space density; and a tolerance, so that we consider data points
    where mu is between mu_(1-tolerance) and mu_(1+tolerance) and I is
    between I_(1-tolerance) and I_(1+tolerance). It then returns a list of
    points, where each "point" represent a selected data point as a tuple
    (epoch, Lstar, I, mu, PSD).
    """
    shape = cdf["PSD"].shape
    nEpochs, nLstars, nPoints = shape
    points = []
    for i in range(nEpochs):
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
                            log = np.log(cdf["PSD"][i, j, k])
                            points.append((epoch, Lstar, I, mu, log))
    return points

def points_into_orbits(OrbTimes, points):
    """
    Takes a list of times and a list of points where each point is of the
    form given by points_from_cdf (epoch, Lstar, I, mu, PSD), and returns a
    list orbit_points such that orbit_points[i] is a list of the points that
    are in orbit i (where the number of the orbit is as defined in find_orbit).
    """
    orbit_points = [[] for i in range(len(OrbTimes) + 1)]
    for point in points:
        orbit_points[find_orbit(OrbTimes, point[0])].append(point)
    return orbit_points

def data_from_orbit_points(ops, L_range, nL, L_tolerance):
    """
    Takes ops (a list of the type that points_into_orbits returns), in other
    words a list of lists of points, where each sublist represents all of
    the points in the i^{th} orbit where each point is of the form
    (epoch, Lstar, I, mu, PSD); L_range, which is a tuple of (L_min, L_max)
    representing the smallest  and largest Lstars that we care about; and
    nL, which is the number of different L values we consider.
    Returns a tuple (Li, ts, F_bars), where Li is the list of Lstars that we
    are considering, ts and F_bars are np.arrays, with one row per orbit,
    and nL columns each of which represents the corresponding L-value.
    If we have data for a given L-value in a given orbit, then the
    corresponding entry in F_bars is the average density based all the
    points from the right orbit within 0.1 of the correct Lstar, and the
    entry of ts is the average of the times from each of these points. If we
    do not have data for an entry, we will instead fill the entry with np.nan.
    """
    ts = []
    U_bars = [] # U = log f
    Li = np.linspace(L_range[0], L_range[1], nL, dtype=float)
    for i in range(len(ops)):
        op = ops[i]
        ts.append([])
        U_bars.append([])
        for L in Li:
            ps = list(filter(lambda p: L - L_tolerance <= p[1]
                                       <= L + L_tolerance, op))
            if len(ps) > 0:
                t = np.average([p[0] for p in ps])
                U = np.mean([p[-1] for p in ps], dtype=float)
            else:
                t = np.nan
                U = np.nan
            ts[-1].append(t)
            U_bars[-1].append(U)
        # plt.plot(Li, F_bars[-1], "ro")
        # plt.show()
    ts = np.array(ts, dtype=float)
    U_bars = np.array(U_bars, dtype=float)
    return (Li, ts, U_bars)

def find_min_max_times(ts):
    """
    The goal of this function is to take a matrix ts of the type returned by
    data_from_orbit_points, and return a tuple (mintime, maxtime) where
    mintime is the earliest time that we have data from all Lstars
    that we care about, and maxtime is the latest.
    """
    mintimes = []
    maxtimes = []
    for i in range(ts.shape[1]):
        times = [t for t in ts[:, i] if not np.isnan(t)]
        mintimes.append(np.min(times, axis=0))
        maxtimes.append(np.max(times, axis=0))
    plt.plot([i for i in range(len(mintimes))], mintimes)
    mintime = max(mintimes)
    maxtime = min(maxtimes)
    return mintime, maxtime

def interpolated_log_PSD(L, t, Li, ts, U_bars):
    """
    Takes a value of L and t; and Li, ts and F_bars (all of the forms
    returned by data_from_orbit_points), and returns the phase space density
    interpolated at that particular value of L and t.
    """
    # So L is between L_m and L_{m+1}, and t is between t_n and t_{n+1}
    DL = (Li[-1] - Li[0]) / (len(Li) - 1)
    m = math.ceil((L-Li[0])/DL) - 1
    if m == -1:
        assert(L == Li[0])
        m += 1
    L_m = Li[m]
    assert(L_m <= L)
    # First we'll consider L_m
    present_indices = [i for i, t in enumerate(ts[:, m]) if not np.isnan(t)]
    times = ts[present_indices, m]
    Us = U_bars[present_indices, m]
    n = -1
    for i, ti in enumerate(times):
        if t <= ti:
            n = i-1
            break
    if n == -1:
        if t == times[0]:
            n = 0
        else:
            raise IndexError("Data not available for this time")
    U_nm, U_n1m = Us[n], Us[n+1]
    t_n, t_n1 = times[n], times[n+1]
    p = (t-t_n)/(t_n1-t_n)
    assert(0 <= p <= 1)
    U_m = U_nm*(1-p) + U_n1m*p
    if L == L_m:
        return U_m
    L_m1 = Li[m+1]
    # Now we'll consider L_{m+1}
    present_indices = [i for i, t in enumerate(ts[:, m+1]) if not np.isnan(t)]
    times = ts[present_indices, m+1]
    Us = U_bars[present_indices, m+1]
    n = -1
    for i, ti in enumerate(times):
        if t <= ti:
            n = i - 1
            break
    if n == -1:
        if t == times[0]:
            n = 0
        else:
            raise IndexError("Data not available for this time")
    U_nm1, U_n1m1 = Us[n], Us[n+1]
    t_n, t_n1 = times[n], times[n+1]
    p = (t - t_n) / (t_n1 - t_n)
    assert(0 <= p <= 1)
    U_m1 =  U_nm1 * (1 - p) + U_n1m1 * p
    # Now we interpolate between L_m and L_m+1
    q = (L-L_m) / (L_m1 - L_m)
    U = U_m * (1-q) + U_m1 * q
    assert not np.isnan(U)
    return U


def complete_log_PSD(Li, ts, F_bars, t_range, nTimes):
    """
    Uses interpolated_PSD to interpolate our phase space density to one at
    regular times. It takes Li, ts, F_bars of the types returned by
    data_from_orbit_points, and a t_range (mintime, maxtime) and nTimes
    which is the number of times we want data from. It returns a tuple
    (times, PSD) where times is a list of nTimes equally spaced points,
    the first of which is mintime, and the last of which is maxtime; and
    PSD is a numpy array such that PSD[i, j] is the (interpolated) phase
    space density at time times[i] and at Lstar Li[j].
    """
    times = np.linspace(t_range[0], t_range[1], nTimes)
    log = np.array([[interpolated_log_PSD(L, t, Li, ts, F_bars) for L in Li]
                    for t in times], dtype=float)
    assert not np.any(np.isnan(log))
    return (times, log)

def combine_CDFs_from_dir(dir_path):
    if dir_path[-1] == "/":
        dir_path = dir_path[:-1]
    files = [file for file in os.listdir(dir_path)
             if file[:11] == "PSD_rbspb_m"]
    cdfs = [pycdf.CDF(dir_path + "/" + file) for file in files]
    cdfs = sorted(cdfs, key = lambda cdf : cdf["OrbTimes"][0])
    combined = {}
    for key in cdfs[0].keys():
        data = np.concatenate([cdf[key] for cdf in cdfs], axis=0)
        if key == "Epoch" or key == "OrbTimes":
            data = np.vectorize(timeToDays)(data)
        combined[key] = data
    return combined

def process_CDF_dictionary(cdf, L_range, DL, Dt, mu_, I_, L_tol):
    nL = round((L_range[-1] - L_range[0]) / DL) + 1
    points = points_from_cdf(cdf, mu_, I_, 0.16)
    OrbTimes = cdf["OrbTimes"]
    orbit_points = points_into_orbits(OrbTimes, points)
    Li, ts, U_bars = data_from_orbit_points(orbit_points, L_range, nL, L_tol)
    t_range = find_min_max_times(ts)
    nT = round((t_range[-1] - t_range[0]) / Dt) + 1
    complete = complete_log_PSD(Li, ts, U_bars, t_range, nT)
    assert not np.any(np.isnan(complete[1]))
    return complete

def initial_from_CDF_dictionary(cdf, L_range, DL, mu_, I_, L_tol):
    nL = round((L_range[-1] - L_range[0]) / DL) + 1
    points = points_from_cdf(cdf, mu_, I_, 0.16)
    OrbTimes = cdf["OrbTimes"]
    orbit_points = points_into_orbits(OrbTimes, points)
    Li, ts, U_bars = data_from_orbit_points(orbit_points, L_range, nL, L_tol)
    t_range = find_min_max_times(ts)
    nT = 2
    times, complete = complete_log_PSD(Li, ts, U_bars, t_range, nT)
    return times[0], complete[0, :]


def boundaries_from_CDF_dictionary(cdf, L_range, DL, Dt, mu_, I_, L_tol):
    nL = round((L_range[-1] - L_range[0]) / DL) + 1
    points = points_from_cdf(cdf, mu_, I_, 0.16)
    OrbTimes = cdf["OrbTimes"]
    orbit_points = points_into_orbits(OrbTimes, points)
    Li, ts, U_bars = data_from_orbit_points(orbit_points, L_range, nL, L_tol)
    t_range = find_min_max_times(ts)
    nT = round((t_range[-1] - t_range[0]) / Dt) + 1
    times, complete = complete_log_PSD(Li, ts, U_bars, t_range, nT)
    t_range, L_range = (times[0], times[-1]), (Li[0], Li[-1])
    left, right = complete[:, 0], complete[:, -1]
    initial = complete[0, :]
    return (t_range, L_range, left, right, initial)

def preprocessing_Kalman(L_range, t_range, log_initial, log_left, log_right,
                         OrbTimes, VAP_logs, VAP_times):
    model_Li = np.linspace(L_range[0], L_range[-1], log_initial.shape[0])
    VAP_Li = np.linspace(L_range[0], L_range[-1], VAP_times.shape[1])
    m, n = VAP_Li.shape[0], model_Li.shape[0]
    H_all_data = [[1 if round(j * (m - 1) / (n - 1)) == i else 0
                   for j in range(n)] for i in range(m)]
    H_all_data = [[Hij/np.sum(Hi) for Hij in Hi] for Hi in H_all_data]
    boundary_times = np.linspace(t_range[0], t_range[-1], log_left.shape[0])
    assert len(OrbTimes) + 1 == len(VAP_logs) == len(VAP_times)
    assert VAP_logs.shape == VAP_times.shape
    orbits = [[OrbTimes[i], OrbTimes[i + 1]] for i in range(len(OrbTimes) - 1)]
    if t_range[0] < OrbTimes[0]:
        orbits = [t_range[0], OrbTimes[0]]
    else:
        VAP_times = np.delete(VAP_times, 0, axis=0)
        VAP_logs = np.delete(VAP_logs, 0, axis=0)

    if t_range[1] > OrbTimes[-1]:
        orbits = [OrbTimes[-1], t_range[1]]
    else:
        VAP_times = np.delete(VAP_times, -1, axis=0)
        VAP_logs = np.delete(VAP_logs, -1, axis=0)

    assert len(orbits) == len(VAP_logs) == len(VAP_times)
    for i, orbit in enumerate(orbits):
        if orbit[1] > t_range[0]: # orbits[i] is first one in
            orbits = orbits[i:]
            VAP_times = VAP_times[i:]
            VAP_logs = VAP_logs[i:]
            break
    for i, orbit in enumerate(orbits):
        if orbit[0] >= t_range[1]: # orbits[i] is first one out
            orbits = orbits[:i]
            VAP_times = VAP_times[:i]
            VAP_logs = VAP_logs[:i]
            break
    for i in range(VAP_times.shape[1]):
        if VAP_times[0, i] < t_range[0]:
            VAP_times[0, i] = np.nan
            VAP_logs[0, i] = np.nan
        if VAP_times[-1, i] > t_range[1]:
            VAP_times[-1, i] = np.nan
            VAP_logs[-1, i] = np.nan
    orbits[0][0] = t_range[0]
    orbits[-1][1] = t_range[1]
    boundary_times = [list(filter(lambda x: orbit[0] <= x[1] <= orbit[1],
                                  enumerate(boundary_times)))
                      for orbit in orbits]


    for i in range(len(orbits) - 1): # duplicate all the boundary points
        boundary_times[i].append(boundary_times[i+1][0])
        boundary_times[i+1].insert(0, boundary_times[i][-2])

    orbit_log_lefts = [[log_left[i[0]] for i in orbit]
                      for orbit in boundary_times]
    orbit_log_rights = [[log_right[i[0]] for i in orbit] for orbit in
                       boundary_times]
    boundary_times = [[t[1] for t in orbit] for orbit in boundary_times]
    assert np.all([orbits[i][0] >= boundary_times[i][0]
                   for i in range(len(orbits))])
    assert np.all([orbits[i][1] <= boundary_times[i][-1]
                   for i in range(len(orbits))])
    return {
        "L_range" : L_range,
        "t_range" : t_range,
        "orbits" : orbits,
        "orbit_boundary_times" : boundary_times,
        "orbit_log_lefts" : orbit_log_lefts,
        "orbit_log_rights" : orbit_log_rights,
        "log_initial" : log_initial.tolist(),
        "VAP_times" : VAP_times.tolist(),
        "VAP_logs" : VAP_logs.tolist(),
        "H" : H_all_data,
        "model_Li" : model_Li.tolist(),
    }

if __name__ == "__main__":
    # for dir_path in ["day"]:
    # for dir_path in ["week", "month", "month2"]:
    for dir_path in ["day", "week", "month", "month2"]:
        for file in ["/diffusion_input.cdf", "/kalman_boundary.cdf",
                     "/kalman_data.cdf"]:
            if file[1:] in os.listdir(dir_path):
                os.remove(dir_path + file)
        print(dir_path)
        data_dict = combine_CDFs_from_dir(dir_path + "/cdfs")
        L_range = (3.1, 5.3)
        DL = 0.01
        # mu_ = 350
        # I_ = 0.1
        mu_ = 700
        I_ = 0.11
        # L_tol = 0.1
        # Dt = 0.01
        # times, log_PSD = process_CDF_dictionary(data_dict, L_range, DL, Dt,
        #                                         mu_,
        #                                         I_, L_tol)
        # print(times[0], times[-1])
        # print("Making diffision_input.cdf")
        # nL = round((L_range[-1] - L_range[0]) / DL) + 1
        # with pycdf.CDF(dir + '/diffusion_input.cdf', '') as outputCDF:
        #     assert not np.any(np.isnan(log_PSD))
        #     assert log_PSD.shape == (len(times), nL)
        #     outputCDF["log_PSD"] = log_PSD
        #     outputCDF["Li"] = np.linspace(L_range[0], L_range[-1], nL)
        #     outputCDF["times"] = times

        initial_DL = 0.01
        Dt = 0.01
        L_tol = 0.1
        boundaries = boundaries_from_CDF_dictionary(data_dict, L_range,
                                                    initial_DL, Dt, mu_,
                                                    I_, L_tol)
        t_range, L_range, log_left, log_right, log_initial = boundaries
        assert not np.isnan(t_range[0])
        assert not np.isnan(t_range[1])
        assert not np.isnan(L_range[0])
        assert not np.isnan(L_range[1])
        assert log_left.shape == log_right.shape
        assert not (np.any(np.isnan(log_left))
                    and np.any(np.isnan(log_right)))

        nL = round((L_range[1] - L_range[0]) / 0.1 + 1)
        DL = (L_range[1] - L_range[0]) / (nL - 1)
        DL = 0.1  # To make it a nice round number rather than 0.099999999999
        points = points_from_cdf(data_dict, mu_, I_, 0.15)
        OrbTimes = data_dict["OrbTimes"]
        orbitPoints = points_into_orbits(OrbTimes, points)
        Li, VAP_times, VAP_logs = data_from_orbit_points(orbitPoints, L_range,
                                                         nL, DL / 2)
        assert VAP_times.shape == VAP_logs.shape == (len(OrbTimes) + 1,
                                                    Li.shape[0])
        assert np.all(np.diff(OrbTimes) >= 0) # Checks OrbTimes is increasing
        kalman_dict = preprocessing_Kalman(L_range, t_range, log_initial,
                                           log_left, log_right, OrbTimes,
                                           VAP_logs, VAP_times)
        with open(dir_path + '/kalman_input.json', "w") as outputJSON:
            json.dump(kalman_dict, outputJSON, indent=4)
