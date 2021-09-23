import numpy as np
from spacepy import pycdf
import matplotlib.pyplot as plt
import os
import datetime
import math
import json

MINTIME = datetime.datetime(datetime.MINYEAR, 1, 1)

def time_to_days(t):
    """
        Takes a time (t) in the form of datetime.datetime and returns the
        number of days since January 1st in datetime.MINYEAR.

            Parameters:

                t (datetime.datetime) : Any time.


            Returns:

                days (float) : The number of days by which t is after January
                1st in datetime.MINYEAR
    """
    delta = t-MINTIME
    days = delta.total_seconds()/86400
    return days

def find_orbit(OrbTimes, t):
    """
        Takes a list OrbTimes of times at which orbits ended, and a time,
        and returns which orbit that time was during. OrbTimes must already be
        sorted before this function is called.
        The way that the numbering works is that if t is before any of the
        OrbTimes, it is counted as being in orbit 0; if it is later than all
        the OrbTimes, it is counted as being in orbit len(OrbTimes); and if
        OrbTimes[i-1] <= t < OrbTimes[i], it is counted as being in
        Orbit i.

            Parameters:

                OrbTimes (list) : a list of floats representing the times at
                the ends of orbits. These must be sorted.

                t (float) : Any time.


            Returns:

                orbit (int) : The orbit that t is in, where orbit 0 is before
                all of the times in OrbTimes; orbit len(OrbTimes) is after all
                of the times in OrbTimes (or equal to the last time); and all
                other orbits i represent the time range
                [OrbTimes[i-1], OrbTimes[i]).
    """
    for i, orb in enumerate(OrbTimes):
        if t < orb:
            return i
    return len(OrbTimes)

def points_from_dict(dictionary, mu_, K_, tolerance):
    """
    Takes a dictionary containing data extracted from our CDF's; a value of
    mu and a value of K; and a tolerance, and extracts all of the points in
    the dictionary that are at mu and K such that
    mu_(1-tolerance) <= mu <= mu_(1+tolerance) and
    K_(1-tolerance) <= K <= K_(1+tolerance)

        Parameters:

            dictionary (dict) : a dictionary of the type returned by
            combine_CDFs_from_dir, in other words a dictionary of the same
            form as the CDF's downloaded from https://rbspgway.jhuapl.edu/psd,
            except that all of the times have been converted into floats.

            mu_ (float) : a float representing the value of mu at which
            we want to model the phase space density.

            K_ (float) : a float representing the value of K at which
            we want to model the phase space density.

            tolerance : a float between zero and one


        Returns:

            points (list) : a list of points, where each point is a tuple of
            the form (time, L*, K, mu, ln(PSD)) (where ln(PSD) is the
            natural log of the phase space density at that point.)
    """
    nEpochs, nLstars, nPoints = dictionary["PSD"].shape
    points = []
    for i in range(nEpochs):
        epoch = dictionary["Epoch"][i]
        for j in range(nLstars):
            for k in range(nPoints):
                Lstar = dictionary["Lstar"][i, j]
                K = dictionary["I"][i, j]
                mu = dictionary["mu"][i, j, k]
                PSD = dictionary["PSD"][i, j, k]
                if (1-tolerance<=mu/mu_<=1+tolerance):
                    if (1-tolerance<=K/K_<=1+tolerance):
                        if PSD > 0 and Lstar > 0:
                            log = np.log(dictionary["PSD"][i, j, k])
                            points.append((epoch, Lstar, K, mu, log))
    return points

def points_into_orbits(OrbTimes, points):
    """
        Takes a list of times and a list of points, and returns a list
        orbit_points such that orbit_points[i] is a list of the points that
        are in orbit i (where the number of the orbit is as defined in
        find_orbit).

            Parameters:

                OrbTimes (list) : a list of floats representing the times at
                the ends of orbits. These must be sorted.

                points (list) : a list of points, where each point is a tuple
                of the form (time, L*, K, mu, ln(PSD)), where ln(PSD) is the
                natural log of the phase space density at that point.


            Returns:

                orbit_points (list) : a list of length (len(OrbTimes)) + 1 such
                that orbit_points[i] is the list of points that are in orbit i
                (as defined in find_orbit, and where each point is of the form
                (time, L*, K, mu, ln(PSD))
    """
    orbit_points = [[] for i in range(len(OrbTimes) + 1)]
    for point in points:
        orbit_points[find_orbit(OrbTimes, point[0])].append(point)
    return orbit_points

def data_from_orbit_points(orbit_points, L_range, n_L, L_tolerance):
    """
    Takes orbit_points, a list of sublists of points, where each sublist is
    a list of points from a particular orbit. We then discretize our L_range
    into n_L equally spaced points, and in each orbit we iterate over these
    L-values and average the log phase space density at all points within
    L_tolerance of our L-value and we store these in a np.array where each
    row represents a different orbit, and each column represents a
    different L-value. We also store the average of the times contributing
    to this point in a different array with the same dimension. Wherever we
    can't find any points to average, we put np.nan in the arrays instead.

        Parameters:

            orbit_points (list) : a list of length (len(OrbTimes)) + 1 such
            that orbit_points[i] is the list of points that are in orbit i
            (as defined in find_orbit, and where each point is of the form
            (time, L*, K, mu, ln(PSD)).

            L_range (tuple) : a tuple (L_0, L_f) where L_0 and L_f are
            floats representing the minimum and maximum values of L that we
            are considering.

            n_L (int) : the number of points into which we want to discretize
            our L_range.

            L_tolerance (float) : the tolerance over which we want to
            average our points.


        Returns:

            output (tuple) : (Ls, ts, U_bars) where:

                Ls (np.array) is the list of L-values into which we've
                discretized our L_range (Ls.shape = (n_L,))

                ts (np.array) is an array of times, where each orbit is
                represented by one row of the array, and each L-value is
                represented by a column (ts.shape = (len(OrbTimes) + 1, n_L))

                U_bars (np.array) is an array of the same shape as ts. Each
                point in the array represents the averaged log of the phase
                space density at that point.
    """
    ts = []
    U_bars = [] # U = log f
    Ls = np.linspace(L_range[0], L_range[1], n_L, dtype=float)
    for i in range(len(orbit_points)):
        op = orbit_points[i]
        ts.append([])
        U_bars.append([])
        for L in Ls:
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
        # plt.plot(Ls, F_bars[-1], "ro")
        # plt.show()
    ts = np.array(ts, dtype=float)
    U_bars = np.array(U_bars, dtype=float)
    return (Ls, ts, U_bars)

def find_min_max_times(ts):
    """
        The goal of this function is to take a matrix ts of the type returned
        by data_from_orbit_points, and return a tuple (mintime, maxtime) where
        mintime is the earliest time such that for all L-values we have a data
        point from that time or earlier, and maxtime is the latest time such
        that for all L-values we have a data point from that time or later.

        Parameters:

                ts (np.array) : an array of times (as returned by
                data_from_orbits, where each orbit is represented by one row of
                the array, and each L-value is represented by a column. If we
                have data in a particlar orbit at a particular L-value, then
                the corresponding point in the array will be a float,
                and otherwise it will be np.nan.


            Returns:

                output (tuple) : (mintime, maxtime) where:

                    mintime (float) is the the earliest time such that for all
                    L-values we have a data point from that time or earlier.

                    maxtime (float) is the latest time such that for all
                    L-values we have a data point from that time or later.

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

def interpolated_log_PSD(L, t, Ls, ts, U_bars):
    """
        Takes a value of L and t; and Ls, ts and F_bars (all of the forms
        returned by data_from_orbit_points), and returns the log phase space
        density interpolated at that particular value of L and t.
        t should be within find_min_max_times(ts), and L should be greater than
        or equal to Ls[0] and less than or equal to Ls[-1].

        Parameters:

                L (float) : the L-value we're interested in

                t (float) : the time we're interested in

                Ls (np.array) : the list of L-values into which we've
                discretized our L_range (Ls.shape = (n_L,))

                ts (np.array) : an array of times, where each orbit is
                represented by one row of the array, and each L-value is
                represented by a column (ts.shape = (ts, n_L))

                U_bars (np.array) : an array of the same shape as ts. Each
                point in the array represents the averaged log of the phase
                space density at that point


            Returns:

                U (float) : the log phase space density linearly interpolated
                from the nearby points.
    """
    # So L is between L_m and L_{m+1}, and t is between t_n and t_{n+1}
    DL = (Ls[-1] - Ls[0]) / (len(Ls) - 1)
    m = math.ceil((L-Ls[0])/DL) - 1
    if m == -1:
        assert(L == Ls[0])
        m += 1
    L_m = Ls[m]
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
    L_m1 = Ls[m+1]
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


def complete_log_PSD(Ls, ts, U_bars, t_range, n_t):
    """
    Uses interpolated_PSD to interpolate our phase space density to one at
    regular times. It takes Ls, ts, F_bars of the types returned by
    data_from_orbit_points, and a t_range (mintime, maxtime) and n_t
    which is the number of times we want data from. It returns a tuple
    (times, PSD) where times is a list of n_t equally spaced points,
    the first of which is mintime, and the last of which is maxtime; and
    PSD is a numpy array such that PSD[i, j] is the (interpolated) phase
    space density at time times[i] and at Lstar Ls[j].

    Parameters:

            Ls (np.array) : the list of L-values into which we've
            discretized our L_range (Ls.shape = (n_L,))

            ts (np.array) : an array of times, where each orbit is
            represented by one row of the array, and each L-value is
            represented by a column (ts.shape = (ts, n_L))

            U_bars (np.array) : an array of the same shape as ts. Each
            point in the array represents the averaged log of the phase
            space density at that point

            t_range (tuple) : a tuple (t_0, t_f) where t_0 and t_f are
            floats representing the earliest and latest times that we
            want to find the log PSD at.

            n_t (int) : the number of times into which we want to discretize
            our t_range

    Returns:

            output (tuple) : (times, log) where:

                times (np.array) is array of times into which we have
                discretized our t_range

                log (np.array) is the interpolated phase space density such
                that log[i, j] is the log phase space density at times[i],
                Ls[j]
    """
    times = np.linspace(t_range[0], t_range[1], n_t)
    log = np.array([[interpolated_log_PSD(L, t, Ls, ts, U_bars) for L in Ls]
                    for t in times], dtype=float)
    assert not np.any(np.isnan(log))
    return (times, log)

def combine_CDFs_from_dir(cdf_path):
    """
        Takes a path to a directory filled with CDF's of the type downloaded
        from https://rbspgway.jhuapl.edu/psd, and finds all the files
        corresponding to CDF's with the PSD's from Van Allen Probe B's
        MagEIS. It then turns these CDF's into dictionaries and combines
        them into one dictionary. It then converts the times in this
        dictionary into floats (in days) and returns this new dictionary.

    Parameters:

            cdf_path (str) : the path of the directory containing our CDF's
            (NOT the directory containing the config file!!!)

    Returns:

            dictionary (dict) : a dictionary consisting of all of the
            dictionaries representing the CDF's put together but with the
            time converted into floats.

    """
    if cdf_path[-1] == "/":
        cdf_path = cdf_path[:-1]
    files = [file for file in os.listdir(cdf_path)
             if file[:11] == "PSD_rbspb_m"]
    cdfs = [pycdf.CDF(cdf_path + "/" + file) for file in files]
    cdfs = sorted(cdfs, key = lambda cdf : cdf["OrbTimes"][0])
    combined = {}
    for key in cdfs[0].keys():
        data = np.concatenate([cdf[key] for cdf in cdfs], axis=0)
        if key == "Epoch" or key == "OrbTimes":
            data = np.vectorize(time_to_days)(data)
        combined[key] = data
    return combined

def process_CDF_dictionary(dictionary, L_range, DL, Dt, mu_, K_, L_tol):
    """
        Takes a dictionary and a set of parameters that we will use in our
        model and then finds a complete log phase space density with those
        parameters.

        Parameters:

                dictionary (dict) : a dictionary consisting of all of the
                dictionaries representing the CDF's put together but with the
                time converted into floats.

                L_range (tuple) : a tuple (L_0, L_f) where L_0 and L_f are
                floats representing the minimum and maximum values of L that we
                are considering.

                DL (float) : represents the distance between points we want
                to use when discretizing our L_range

                Dt (float) : represents the distance between points we want
                to use when discretizing our t_range (we find the t_range
                automatically using find_min_max_times)

                mu_ (float) : a float representing the value of mu at which
                we want to model the phase space density.

                K_ (float) : a float representing the value of K at which
                we want to model the phase space density.

                L_tol : a float to be used as the tolerance in L over which we
                average our log densities.


        Returns:

                complete (np.array) : the interpolated phase space density such
                that log[i, j] is the log phase space density at times[i],
                Ls[j]
    """
    n_L = round((L_range[-1] - L_range[0]) / DL) + 1
    points = points_from_dict(dictionary, mu_, K_, 0.15)
    OrbTimes = dictionary["OrbTimes"]
    orbit_points = points_into_orbits(OrbTimes, points)
    Ls, ts, U_bars = data_from_orbit_points(orbit_points, L_range, n_L, L_tol)
    t_range = find_min_max_times(ts)
    n_t = round((t_range[-1] - t_range[0]) / Dt) + 1
    complete = complete_log_PSD(Ls, ts, U_bars, t_range, n_t)
    assert not np.any(np.isnan(complete[1]))
    return complete


def process_directory(dir_path):
    """
        Takes a directory path, and returns a dictionary to be put into JSON
        containing all the data to be fed into models.py. The directory must
        contain a config file in "config.json" and a folder "cdfs" that
        contains all of the CDF's that we want to process.

        Parameters:

                dir_path (dict) : the path to a directory containing our
                config file and a subfolder "cdfs" containing all of our CDF's


        Returns:

                models_dict (dict) : a dictionary to be loaded into a JSON
                file and used in models.py. It will be of the form:

                    {
                        "L_range": L_range,
                        "t_range": t_range,
                        "DL": diffusion_DL,
                        "Dt": Dt,
                        "mu": mu,
                        "K": K,
                        "L_tol": L_tol,
                        "PSD": PSD
                        "VAP_points": VAP_points
                    }

                where:
                    L_range is a tuple (L_0, L_f) where L_0 and L_f are
                    floats representing the minimum and maximum values of L
                    that we are considering;

                    t_range is a tuple (t_0, t_f) where t_0 and t_f are
                    floats representing the earliest and latest time that we
                    are considering;

                    DL (float) : represents the distance between points we want
                    to use when discretizing our L_range in our model

                    Dt (float) : represents the distance between points we want
                    to use when discretizing our t_range in our model

                    mu (float) : a float representing the value of mu at which
                    we want to model the phase space density.

                    K (float) : a float representing the value of K at which
                    we want to model the phase space density.

                    L_tol (float) : a float that was used as the tolerance
                    in L over which we averaged our log densities.

                    PSD (np.array) : the phase space density interpolated
                    from the Van Allen Probe data at the same resolution
                    that we want to use in the model. Each row represents a
                    different time, and each column represents a different
                    L-value

                    VAP_points (np.array) : We discretize our L-range such
                    that the difference between points is 2 * L_tol.
                    In each orbit, for each L in our discretized L-range we
                    average our phase space density at all points within
                    L_tol of our L and attach the average of their times to
                    it. We then snap each of these points to our set of
                    discrete times, and put these into a np.array where each
                    row represents a time and each column represents a
                    L-value. Where we don't have a point, we put np.nan in
                    the array instead.

    """
    data_dict = combine_CDFs_from_dir(dir_path + "/cdfs")
    with open(dir_path + "/config.json", "r") as f:
        config = json.load(f)
    L_range = config["L_range"]
    diffusion_DL = config["diffusion_DL"]
    Dt = config["Dt"]
    mu = config["mu"]
    K = config["K"]
    VAP_DL = config["obs_DL"]
    VAP_n_L = round((L_range[1] - L_range[0]) / VAP_DL + 1)
    VAP_DL = (L_range[1] - L_range[0]) / (VAP_n_L - 1)
    L_tol = VAP_DL / 2
    model_times, log_PSD = process_CDF_dictionary(data_dict, L_range,
                                                  diffusion_DL, Dt, mu,
                                                  K, L_tol)

    t_range = (model_times[0], model_times[-1])
    model_n_t = model_times.shape[0]
    model_n_L = log_PSD.shape[1]
    assert (model_n_t, model_n_L) == log_PSD.shape

    points = points_from_dict(data_dict, mu, K, 0.15)
    OrbTimes = data_dict["OrbTimes"]
    orbitPoints = points_into_orbits(OrbTimes, points)
    Ls, VAP_times, VAP_logs = data_from_orbit_points(orbitPoints, L_range,
                                                     VAP_n_L, L_tol)

    VAP_times = (VAP_times - t_range[0]) / (t_range[1] - t_range[0])
    VAP_times *= model_n_t - 1
    VAP_times = np.round(VAP_times).astype(int)
    VAP_times[VAP_times >= model_n_t] = model_n_t
    VAP_times[VAP_times < 0] = model_n_t
    VAP_times[np.isnan(VAP_times)] = model_n_t
    VAP_times = VAP_times.astype(int)

    VAP_times = [[(VAP_times[:, j] == i).nonzero()
                  for j in range(VAP_n_L)] for i in range(model_n_t)]
    VAP_points = [[np.exp(np.average(VAP_logs[:, j][VAP_times[i][j]]))
                   for j in range(VAP_n_L)] for i in range(model_n_t)]

    models_dict = {
        "L_range": L_range,
        "t_range": t_range,
        "DL": diffusion_DL,
        "Dt": Dt,
        "mu": mu,
        "I": K,
        "L_tol": L_tol,
        "PSD": np.exp(log_PSD).tolist(),
        "VAP_points": VAP_points
    }
    return models_dict

if __name__ == "__main__":
    dir_path = ["example"]
    models_dict = process_directory(dir_path)
    with open(dir_path + '/models_input.json', "w") as inputJSON:
        json.dump(models_dict, inputJSON, indent=4)
    print("Complete!")

