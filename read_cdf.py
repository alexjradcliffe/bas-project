import numpy as np
from spacepy import pycdf
import matplotlib.pyplot as plt
import os
import datetime
import math

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
                            points.append((epoch, Lstar, I, mu, PSD))
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

def data_from_orbit_points(ops, L_range, nL):
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
    F_bars = []
    referenceDate = datetime.datetime(2000, 1, 1, 0, 0, 0, 0)
    Li = np.linspace(L_range[0], L_range[1], nL)
    for i in range(len(ops)):
        op = ops[i]
        # # Lstar = [point[1] for point in op]
        # # PSD = [point[-1] for point in op]
        # # plt.plot(Lstar, PSD, "ro")
        # # plt.show()
        ts.append([])
        F_bars.append([])
        for L in Li:
            ps = list(filter(lambda p: L - 0.1 <= p[1] <= L + 0.1, op))
            if len(ps) > 0:
                t = referenceDate + sum([p[0] - referenceDate for p in ps], datetime.timedelta()) / len(ps)
                F = np.mean([p[-1] for p in ps])
            else:
                t = np.nan
                F = np.nan
            ts[-1].append(t)
            F_bars[-1].append(F)
        # plt.plot(Li, F_bars[-1], "ro")
        # plt.show()
    ts = np.array(ts)
    F_bars = np.array(F_bars)
    return (Li, ts, F_bars)

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
        times = [t for t in ts[:, i] if isinstance(t, datetime.datetime)]
        mintimes.append(np.min(times, axis=0))
        maxtimes.append(np.max(times, axis=0))
    plt.plot([i for i in range(len(mintimes))], mintimes)
    mintime = max(mintimes)
    maxtime = min(maxtimes)
    # print(mintimes)
    # print(np.argmax(mintimes))
    # print(mintime)
    # print(5/0)
    return mintime, maxtime

def time_linspace(mintime, maxtime, nTimes):
    """
    A version of np.linspace that works with datetime.datetimes instead of
    numbers. This returns a list of nTimes times the first of which is
    mintime; the last of which is maxtime, and the rest are equally spaced
    between them.
    """
    timespan = maxtime - mintime
    Dt = timespan / (nTimes - 1)
    times = [mintime + Dt * i for i in range(nTimes)]
    times[-1] = maxtime
    return times

def interpolated_PSD(L, t, Li, ts, F_bars):
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
    present_indices = [i for i, t in enumerate(ts[:, m])
                       if isinstance(t, datetime.datetime)]
    times = ts[present_indices, m]
    Fs = F_bars[present_indices, m]
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
    F_nm, F_n1m = Fs[n], Fs[n+1]
    t_n, t_n1 = times[n], times[n+1]
    p = (t-t_n)/(t_n1-t_n)
    assert(0 <= p <= 1)
    F_m = F_nm**(1-p) * F_n1m**p
    if L == L_m:
        return F_m
    L_m1 = Li[m+1]
    # Now we'll consider L_{m+1}
    present_indices = [i for i, t in enumerate(ts[:, m+1])
                       if isinstance(t, datetime.datetime)]
    times = ts[present_indices, m+1]
    Fs = F_bars[present_indices, m+1]
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
    F_nm1, F_n1m1 = Fs[n], Fs[n+1]
    t_n, t_n1 = times[n], times[n+1]
    p = (t - t_n) / (t_n1 - t_n)
    assert(0 <= p <= 1)
    F_m1 =  F_nm1 ** (1 - p) * F_n1m1 ** p
    # Now we interpolate between L_m and L_m+1
    q = (L-L_m) / (L_m1 - L_m)
    F = F_m ** (1-q) * F_m1 ** q
    return F


def complete_PSD(Li, ts, F_bars, t_range, nTimes):
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
    times = time_linspace(t_range[0], t_range[1], nTimes)
    PSD = np.array([[interpolated_PSD(L, t, Li, ts, F_bars) for L in Li]
                    for t in times])
    return (times, PSD)

def process_multiple_CDFs(dir_path, L_range, DL, Dt, mu_, I_):
    nL = int((L_range[-1] -L_range[0]) / DL) + 1
    if dir_path[-1] == "/":
        dir_path = dir_path[:-1]
    files = [file for file in os.listdir(dir_path)
             if file[:11] == "PSD_rbspb_m"]
    cdfs = [pycdf.CDF(dir_path + "/" + file) for file in files]
    total = {}
    for key in cdfs[0].keys():
        total[key] = np.concatenate([cdf[key] for cdf in cdfs], axis=0)
    points = points_from_cdf(total, mu_, I_, 0.16)
    OrbTimes = total["OrbTimes"]
    orbit_points = points_into_orbits(OrbTimes, points)
    Li, ts, F_bars = data_from_orbit_points(orbit_points, L_range, nL)
    t_range = find_min_max_times(ts)
    nT = int((t_range[-1] - t_range[0]) / Dt) + 1
    return complete_PSD(Li, ts, F_bars, t_range, nT)

# if __name__ == "__main__":
    # # cdf_path = '20170908/PSD_2017251.cdf'
    # cdf_path = '20171226/PSD_2017360.cdf'
    # L_range = (3.1, 5.0)
    # nL = 101
    # nT = 101
    #
    # # print(datetime.datetime(2017, 9, 6, 12, 0)) # 2017-09-06 12:00:00
    # # print(Li[46], Li[47]) # 3.982828282828283 4.002020202020202
    # # print(ts[3,46], ts[3,47], ts[4,46], ts[4,47]) # 2017-09-06 11:24:23.207547 2017-09-06 11:25:52.105263 2017-09-06 17:09:04.020619 2017-09-06 17:08:07.894737
    # # print(interpolated_PSD(4, datetime.datetime(2017, 9, 6, 12, 0), Li, ts, F_bars))
    # # print(F_bars[3,46], F_bars[3,47], F_bars[4,46], F_bars[4,47])
    #
    # # print(complete_PSD(Li, ts, F_bars, t_range, nT))
    #
    # # cdf = pycdf.CDF('sep2017/PSD/PSD_rbspb_mageis_2017249.cdf').copy()
    #
    # times, PSD = PSD_from_CDF(cdf_path, L_range, nL, nT)
    # # outputCDF = pycdf.CDF('20170908/output.cdf', '')
    # outputCDF = pycdf.CDF('20171226/output.cdf', '')
    # outputCDF["PSD"] = PSD
    # outputCDF["Li"] = np.linspace(L_range[0], L_range[-1], nL)
    # outputCDF["times"] = times
    # outputCDF.close()

if __name__ == "__main__":
    # for dir in ["day", "week", "month", "month2"]:
    for dir in ["month"]:
        L_range = (3.1, 5.3)
        DL = 0.001
        Dt = datetime.timedelta(minutes=15)

        # print(datetime.datetime(2017, 9, 6, 12, 0)) # 2017-09-06 12:00:00
        # print(Li[46], Li[47]) # 3.982828282828283 4.002020202020202
        # print(ts[3,46], ts[3,47], ts[4,46], ts[4,47]) # 2017-09-06 11:24:23.207547 2017-09-06 11:25:52.105263 2017-09-06 17:09:04.020619 2017-09-06 17:08:07.894737
        # print(interpolated_PSD(4, datetime.datetime(2017, 9, 6, 12, 0), Li, ts, F_bars))
        # print(F_bars[3,46], F_bars[3,47], F_bars[4,46], F_bars[4,47])

        # print(complete_PSD(Li, ts, F_bars, t_range, nT))

        # cdf = pycdf.CDF('sep2017/PSD/PSD_rbspb_mageis_2017249.cdf').copy()
        mu_ = 350
        I_ = 0.1
        mu_ = 700
        I_ = 0.11
        times,PSD = process_multiple_CDFs(dir + "/cdfs", L_range, DL, Dt, mu_,
                                          I_)
        print(times[0], times[-1])
        # outputCDF = pycdf.CDF('20170908/output.cdf', '')
        # outputCDF = pycdf.CDF('week/week_output.cdf', '')
        outputCDF = pycdf.CDF(dir + '/output.cdf', '')
        outputCDF["PSD"] = PSD
        nL = int((L_range[-1] - L_range[0]) / DL) + 1
        outputCDF["Li"] = np.linspace(L_range[0], L_range[-1], nL)
        outputCDF["times"] = times
        outputCDF.close()
