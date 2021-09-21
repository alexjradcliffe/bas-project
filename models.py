import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import math
import datetime
import json
from read_Kp import read_Kp
from matplotlib.colors import LogNorm
random.seed(101)

MINTIME = datetime.datetime(datetime.MINYEAR, 1, 1)

def perturb_Kp_data(Kp_data):
    """
    Takes a dictionary Kp_data, and returns the dictionary, but with each of
    the Kp values perturbed by adding eps * Kp where epsilon is taken from a
    normal distribution with mean 0 and s.d. 0.5

        Parameters:
            Kp_data (dict) : A dictionary like (for example)
                {
                    15.81 : 2.3,
                    15.89 : 2.1,
                    16.00 : 2.9
                }
                where [15.81, 15.89, 16.00] is a list of times in days,
                and the Kp values at those times are [2.3, 2.1, 2.9].

        Returns:
            perturbed (dict) : A dictionary like (for example)
                {
                    15.81 : 0.7743937333537971,
                    15.89 : 3.2544210625730576,
                    16.00 : 2.1143269771870052
                }
                where [15.81, 15.89, 16.00] is a list of times in days,
                and the Kp values at those times are
                [
                    0.7743937333537971,
                    3.2544210625730576,
                    2.1143269771870052
                ].
    """
    # perturbed = {
    #     t : Kp * random.gauss(1, 0.5) for t, Kp in Kp_data.items()
    # }
    perturbed = {
        t: Kp * random.gauss(1, 0.5) for t, Kp in Kp_data.items()
    }
    # perturbed = {
    #     t: Kp for t, Kp in Kp_data.items()
    # }
    return perturbed

def perturb_log_initial(initial):
    """
    Takes a 1D numpy array of floats, and adds different E_i's to each item
    in the list/array, where the E_i's are sampled from N(0, 0.3).
    """
    perturbed = np.array([U + random.gauss(0, 0.3) for U in initial])
    return perturbed

def diffusion_step(Kpt, D_LL, Li, Lj, Dt, tau, initial, L, R):
    """
    Implements one time step of the finite difference scheme to solve the
    diffusion equation

        Parameters:
            Kpt (float) : the Kp value at the time of the step.

            D_LL (function) : A function D_LL(L, Kpt) that takes a float
            value L and Kpt (a Kp-index) and returns a diffusion coefficient.

            Li (np.array, dtype=float) : the L values that we are considering
            in an array of shape (nL,).
            They should be equally spaced across the range of L-values that we
            are considering.

            Lj (np.array, dtype=float) : the values halfway between the
            values of Li in an array of shape (nL-1).
            For example, if Li were [1, 3, 5, 7], Lj would be [2, 4, 6].

            Dt (float) : the time-step in the finite difference scheme.

            tau (function) : a function tau(L, Kpt) of L and Kp used of the
            loss term.

            initial (np.array, dtype=float) : F at the start of the diffusion
            step, in other words initial[i] = F(t0, Li[i]), where t0 is the
            time of the start of the timestep

            L (float): our left boundary condition, in other words f(t1,
            Li[0]) where t1=t0+Dt is the end time of the update

            R (float): our rightt boundary condition, in other words f(t1,
            Li[0]) where t1=t0+Dt is the end time of the update step.

        Returns:
            final (np.array, dtype=float) : F at the end of the diffusion
            step as calculated with the finite difference scheme, in other
            words final[i] = F(t1, Li[i]).
    """
    nL = initial.shape[0]
    DL = Li[1] - Li[0]
    D_LLj = np.array([(D_LL(Li[i], Kpt) + D_LL(Li[i + 1], Kpt)) / 2
                      for i in range(nL - 1)], dtype=float)
    assert not np.any(np.isinf(D_LLj))
    # D_LLj[i] = D_LL_{i+1/2} = D_LL(Kpt, Lj[i])
    taui = np.array([tau(L, Kpt) for L in Li], dtype=float)
    Xi = np.array([-Dt * Li[i + 1] ** 2 * D_LLj[i] / (DL ** 2 * Lj[i] ** 2)
                   for i in range(nL - 2)], dtype=float)
    Yi = np.array([1 + Dt / taui[i] + (Dt * Li[i + 1] ** 2 / DL ** 2)
                   * (D_LLj[i] / Lj[i] ** 2 + D_LLj[i + 1] / Lj[i + 1] ** 2)
                   for i in range(nL - 2)], dtype=float)
    Zi = np.array([-Dt * Li[i + 1] ** 2 * D_LLj[i + 1]
                   / (DL ** 2 * Lj[i + 1] ** 2)
                   for i in range(nL - 2)], dtype=float)
    Ab = np.array([np.concatenate((np.zeros(2, dtype=float), Zi)),
                   np.concatenate((np.zeros(1, dtype=float), Yi,
                                   np.zeros(1, dtype=float))),
                   np.concatenate((Xi, np.zeros(2, dtype=float)))],
                  dtype=float)
    Ab = Ab[:, 1:-1]
    Ab[0, 0] = 0
    Ab[-1, -1] = 0
    # To be used in solve_banded
    y = initial[1:-1].copy()
    y[0] -= Xi[0] * L
    y[-1] -= Zi[-1] * R
    final = [L] + scipy.linalg.solve_banded((1, 1), Ab, y).tolist() + [R]
    final = np.array(final)
    assert final.shape == initial.shape
    return final

def solve_diffusion(L_range, t_range, initial, uL, uR, Kp_data, D_LL, tau):
    """
    Takes some input parameters, and uses an implicit scheme to solve the
    diffusion equation

        dF/dT = L^2 d/dL (D_LL/L^2 DF/DL) - F/tau(L)

    over a range of L-values given by L_range and a range of t_values given
    by t_range.
    We use initial to give an initial state of the system; uL and uR and
    functions of time giving the boundary conditions at low and high
    L-values respectively; Kp_data to specify the Kp indices at different
    times.
    We then supply both D_LL and tau as functions of Kp and L to put into
    the diffusion equation.

        Parameters:
            L_range (tuple) : A tuple like (3.1, 5.3) where 3.1 is the
            minimum L-value we want to consider and 5.3 is the maximum.

            t_range (tuple) : A tuple like (15.1, 17.3) where 15.1 is the
            minimum t-value we want to consider and 17.3 is the maximum.
            They should both be floats.

            initial (np.array) : a numpy array of initial values for the
            system. These are assumed to come from equally spaced L points
            in L_range (i.e. Li = np.linspace(L_range[0], L_range[1], nL),
            such that initial = [F(t=0, L) for L in Li]).
            Its length (nT) is the number of L-values that the model will
            consider

            uL (np.array) : a numpy array of left (at L=L_range[0]) boundary
            values for the system. These are assumed to come from equally
            spaced times in t_range (i.e. ti = np.linspace(t_range[0],
            t_range[1], nT), such that uL = [F(t, L=L_0) for t in ti]).
            Its length (nT) is the number of time steps that the model will
            take (including the endpoints), and must be the same as the length
            of uR.

            uR (np.array) : a numpy array of left (at L=L_range[1]) boundary
            values for the system. These are assumed to come from equally
            spaced times in t_range (i.e. ti = np.linspace(t_range[0],
            t_range[1], nT), such that uR = [F(t, L=L_1) for t in ti]).
            Its length (nT) is the number of time steps that the model will
            take (including the endpoints), and must be the same as the length
            of uL.

            Kp_data (dict) : A dictionary like (for example)
                {
                    15.81 : 2.3,
                    15.89 : 2.1,
                    16.00 : 2.9
                }
                where [15.81, 15.89, 16.00] is a list of times in days,
                and the Kp values at those times are [2.3, 2.1, 2.9].
                It must have a value at a time earlier than t_range[0],
                and a value at a time later than t_range[1]

            D_LL (function) : A function D_LL(L, Kpt) that takes a float
            value L and Kpt (a Kp-index) and returns a diffusion coefficient.

            tau (function) : A function D_LL(L, Kpt) that takes a float
            value L and Kpt (a Kp-index) and returns a diffusion coefficient.

        Returns:
            F_final (np.array) : An array of shape (nT, nL) which represents
            the phase space density at all of the times in
                ti = np.linspace(t_range[0], t_range[1], nT)
            and all of the L-values in
                Li = np.linspace(L_range[0], L_range[1], nL).
            In other words, F_final[i, j]=F(ti[i], Li[j]).
    """
    nL = initial.shape[0]
    nT = uL.shape[0]
    assert uR.shape == (nT,)
    Kp_times = sorted(Kp_data.keys())
    def Kp(Kp_data, t):
        """
        t in days
        """
        if t == Kp_times[-1]:
            return Kp_data[Kp_times[-1]]
        assert Kp_times[0] <= t <= Kp_times[-1]
        for i, time in enumerate(Kp_times):
            if t < time:
                t1 = Kp_times[i-1]
                Kp1 = Kp_data[t1]
                t2 = time
                Kp2 = Kp_data[t2]
                break
        d = (t-t1)/(t2-t1)
        value = Kp1 * (1-d) + Kp2 * d
        return value
    Li = np.linspace(L_range[0], L_range[1], num=nL)
    times = np.linspace(t_range[0], t_range[1], num=nT)
    DL = (L_range[1] - L_range[0])/(nL-1)
    Dt = (t_range[1] - t_range[0])/(nT-1)
    assert(not np.any(np.isnan(initial)))
    Lj = np.array([(Li[i]+Li[i+1])/2 for i in range(nL-1)], dtype=float)
    # Lj[i] = L_{i+1/2}
    F = [initial]
    for j, t in enumerate(times[1:]):
        # print(i)
        i = j + 1
        Kpt = Kp(Kp_data, t)
        initial = diffusion_step(Kpt, D_LL, Li, Lj, Dt, tau, initial, uL[i],
                              uR[i])
        F.append(initial)
    return np.array(F)

def kalman(L_range, t_range, initial, uL, uR, Kp_data, D_LL, tau, VAP_data,
           nRuns):
    """
    Takes some input parameters, and uses an implicit scheme to solve the
    diffusion equation

        dF/dT = L^2 d/dL (D_LL/L^2 DF/DL) - F/tau(L)

    over a range of L-values given by L_range and a range of t_values given
    by t_range.
    We use initial to give an initial state of the system; uL and uR and
    functions of time giving the boundary conditions at low and high
    L-values respectively; Kp_data to specify the Kp indices at different
    times.
    We then supply both D_LL and tau as functions of Kp and L to put into
    the diffusion equation.

        Parameters:
            L_range (tuple) : A tuple like (3.1, 5.3) where 3.1 is the
            minimum L-value we want to consider and 5.3 is the maximum.

            t_range (tuple) : A tuple like (15.1, 17.3) where 15.1 is the
            minimum t-value we want to consider and 17.3 is the maximum.
            They should both be floats.

            initial (np.array) : a numpy array of initial values for the
            system. These are assumed to come from equally spaced L points
            in L_range (i.e. Li = np.linspace(L_range[0], L_range[1], nL),
            such that initial = [F(t=0, L) for L in Li]).
            Its length (nT) is the number of L-values that the model will
            consider

            uL (np.array) : a numpy array of left (at L=L_range[0]) boundary
            values for the system. These are assumed to come from equally
            spaced times in t_range (i.e. ti = np.linspace(t_range[0],
            t_range[1], nT), such that uL = [F(t, L=L_0) for t in ti]).
            Its length (nT) is the number of time steps that the model will
            take (including the endpoints), and must be the same as the length
            of uR.

            uR (np.array) : a numpy array of left (at L=L_range[1]) boundary
            values for the system. These are assumed to come from equally
            spaced times in t_range (i.e. ti = np.linspace(t_range[0],
            t_range[1], nT), such that uR = [F(t, L=L_1) for t in ti]).
            Its length (nT) is the number of time steps that the model will
            take (including the endpoints), and must be the same as the length
            of uL.

            Kp_data (dict) : A dictionary like (for example)
                {
                    15.81 : 2.3,
                    15.89 : 2.1,
                    16.00 : 2.9
                }
                where [15.81, 15.89, 16.00] is a list of times in days,
                and the Kp values at those times are [2.3, 2.1, 2.9].
                It must have a value at a time earlier than t_range[0],
                and a value at a time later than t_range[1]

            D_LL (function) : A function D_LL(L, Kpt) that takes a float
            value L and Kpt (a Kp-index) and returns a diffusion coefficient.

            tau (function) : A function D_LL(L, Kpt) that takes a float
            value L and Kpt (a Kp-index) and returns a diffusion coefficient.

        Returns:
            F_final (np.array) : An array of shape (nT, nL) which represents
            the phase space density at all of the times in
                ti = np.linspace(t_range[0], t_range[1], nT)
            and all of the L-values in
                Li = np.linspace(L_range[0], L_range[1], nL).
            In other words, F_final[i, j]=F(ti[i], Li[j]).
    """
    nT, nL_VAP = VAP_data.shape
    nL_model = initial.shape[0]
    assert uL.shape == uR.shape == (nT,)
    Kp_times = sorted(Kp_data.keys())
    def Kp(Kp_data, t):
        """
        t in days
        """
        if t == Kp_times[-1]:
            return Kp_data[Kp_times[-1]]
        assert Kp_times[0] <= t <= Kp_times[-1]
        for i, time in enumerate(Kp_times):
            if t < time:
                t1 = Kp_times[i-1]
                Kp1 = Kp_data[t1]
                t2 = time
                Kp2 = Kp_data[t2]
                break
        d = (t-t1)/(t2-t1)
        value = Kp1 * (1-d) + Kp2 * d
        return value
    Li = np.linspace(L_range[0], L_range[1], num=nL_model)
    times = np.linspace(t_range[0], t_range[1], num=nT)
    DL = (L_range[1] - L_range[0])/(nL_model-1)
    Dt = (t_range[1] - t_range[0])/(nT-1)
    assert(not np.any(np.isnan(initial)))
    Lj = np.array([(Li[i]+Li[i+1])/2 for i in range(nL_model-1)], dtype=float)
    # Lj[i] = L_{i+1/2}
    F = [initial]
    runs_Kp_data = [perturb_Kp_data(Kp_data) for i in range(nRuns)]
    log_initial = np.log(initial)
    runs_log_initial = [perturb_log_initial(log_initial) for i in range(nRuns)]
    H = np.zeros((nL_VAP, nL_model))
    for j in range(nL_model):
        i = round(j * (nL_VAP - 1)/(nL_model-1))
        H[i, j] = 1
    G = H.copy()
    for i in range(nL_VAP):
        H[i, :] = H[i, :] / np.sum(H[i, :])
    total_innovation = np.zeros(nL_VAP)
    no_innovations = np.zeros(nL_VAP)
    for j, t in enumerate(times[1:]):
        i = j + 1
        VAP_points = VAP_data[i]
        # print(VAP_points)
        print(str(i) + "/" + str(len(times) - 1))
        avg_innovation = np.zeros(nL_model)
        time_VAP_data = VAP_data[i]
        P_f = np.cov(np.transpose(runs_log_initial))
        print("Sum of variances:", np.trace(P_f))
        # print(time_VAP_data)
        Kps = [Kp(runs_Kp_data[run], t) for run in range(nRuns)]
        print("Kps:", Kps)
        print("max(Kps)", np.max(Kps))
        innovations_at_time = np.zeros(nL_VAP)
        no_innovations_at_time = np.zeros(nL_VAP)
        log_predicteds = []
        for run in range(nRuns):
            Kp_data = runs_Kp_data[run]
            Kpt = Kp(Kp_data, t)
            log_initial = runs_log_initial[run]
            initial = np.exp(log_initial)
            updated = diffusion_step(Kpt, D_LL, Li, Lj, Dt, tau, initial,
                                     uL[i], uR[i])
            log_updated = np.log(updated)
            log_predicteds.append(log_updated)

        for n, VAP_point in enumerate(time_VAP_data):
            if not np.isnan(VAP_point):
                for run in range(nRuns):
                    log_predicted = log_predicteds[run]
                    log_VAP_point = np.log(VAP_point)
                    Hn = H[n, :]
                    log_model_point = Hn @ log_predicted
                    innovation = log_VAP_point - log_model_point
                    # Gn = G[n, :]
                    R = 0.5
                    K = P_f @ Hn.transpose() / (Hn @ P_f @ Hn.transpose() + R)
                    # print(innovation)
                    # print("K", K)
                    log_predicted += K * innovation
                    runs_log_initial[run] = log_predicted
        updated_avg = np.exp(np.average(runs_log_initial, axis=0))
        print("max", np.max(runs_log_initial))
        print("min", np.min(runs_log_initial))
        F.append(updated_avg)
    print(total_innovation, no_innovations)
    return np.array(F), total_innovation/no_innovations

def timeToDays(t):
    """
    Takes a time (t) in the form of datetime.datetime and returns the (float)
    number of days + 1 since midnight on 1st January of that year (jan1,
    provided as a datetime.datetime.
    """
    delta = t-MINTIME
    return delta.total_seconds()/86400

if __name__ == "__main__":
    assert timeToDays(MINTIME) == 0
    # for dir_path in ["day", "week", "month2"]:
    for dir_path in ["day", "week", "month", "month2"]:
    # for dir_path in ["month", "month2"]:
    # for dir_path in ["week"]:
        Kp_data = read_Kp(dir_path + "/Kp_data.lst")
        Kp_data = {timeToDays(t): Kp for t, Kp in Kp_data.items()}
        with open(dir_path + '/models_input.json', "r") as inputJSON:
            models_data = json.load(inputJSON)
        L_range = models_data["L_range"]
        t_range = models_data["t_range"]
        print(t_range, min(Kp_data.keys()), max(Kp_data.keys()))
        VAP_PSD = np.array(models_data["PSD"])
        initial = VAP_PSD[0, :]
        uL = VAP_PSD[:, 0]
        uR = VAP_PSD[:, -1]
        def tau(L, Kpt):
            return 3 / Kpt if Kpt != 0 else np.inf

        def D_LL(L, Kpt):
            return (10**(0.506*Kpt-9.325))*L**(10)

        diffusion_PSD = solve_diffusion(L_range, t_range, initial, uL, uR,
                                        Kp_data, D_LL, tau)
        print(diffusion_PSD.shape)
        print(VAP_PSD.shape)
        print(np.average(diffusion_PSD))
        print(np.average(VAP_PSD))
        print(diffusion_PSD)

        VAP_points = np.array(models_data["VAP_points"])
        with open(dir_path + "/config.json", "r") as f:
            config = json.load(f)
        nRuns = config["nRuns"]
        print(nRuns, "runs")
        kalman_output, innovation = kalman(L_range, t_range, initial, uL, uR,
                                           Kp_data, D_LL, tau, VAP_points,
                                           nRuns)

        output_dict = {
            "diffusion_output" : diffusion_PSD.tolist(),
            "kalman_output" : kalman_output.tolist(),
            "innovation" : innovation.tolist()
        }
        with open(dir_path + '/models_output.json', "w") as outputJSON:
            json.dump(output_dict, outputJSON)
