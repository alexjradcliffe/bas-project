import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import math
import datetime
import json
from read_Kp import read_Kp
from matplotlib.colors import LogNorm
random.seed(100)

MINTIME = datetime.datetime(datetime.MINYEAR, 1, 1)

def perturb_Kp_data(Kp_data):
    perturbed = {
        t : min(0.1, Kp) * random.gauss(1, 0.5) for t, Kp in Kp_data.items()
    }
    return perturbed

def solve_diffusion(L_range, t_range, nL, nT, initial, D_LL, tau, uL, uR,
                    Kp_data):
    """
    PDE is $\frac{dF}{dt}
    =L^2\frac{d}{dL}\left(\frac{1}{L^2}D_{LL}\frac{dF}{dL}\right)
    -\frac{F}{tau(L)}$
    L_range is the range of L that we are considering (a tuple)
    t_range is the range of t that we are considering (a tuple)
    initial condition $F(x, t_{min})=f0(x)$
    boundary conditions are $F(x_{min}, t)=F(x_{min}, t_{min})$;
                            $F(x_{max}, t)=F(x_{max}, t_{min})$
    """
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
    initial_Li = np.linspace(L_range[0], L_range[1], num=initial.shape[0])
    def f0(L):
        return np.exp(interpolate1D(initial_Li, np.log(initial), L))
    Li = np.linspace(L_range[0], L_range[1], num=nL)
    times = np.linspace(t_range[0], t_range[1], num=nT)
    DL = (L_range[1] - L_range[0])/(nL-1)
    Dt = (t_range[1] - t_range[0])/(nT-1)
    initial = np.array([f0(L) for L in Li[1 : -1]], dtype=float)
    uLt = [uL(t) for t in times] # left boundary condition
    uRt = [uR(t) for t in times] # right boundary condition
    Lj = np.array([(Li[i]+Li[i+1])/2 for i in range(nL-1)], dtype=float)
    # Lj[i] = L_{i+1/2}
    ui = np.zeros((nL - 2, nT), dtype=float) #excludes x=0 and x=1
    ui[:, 0] = initial
    for i, t in enumerate(times[1:]):
        i += 1
        Kpt = Kp(Kp_data, t)
        D_LLj = np.array([(D_LL(Li[i], Kpt) + D_LL(Li[i + 1], Kpt)) / 2
                          for i in range(nL-1)], dtype=float)
        assert not np.any(np.isinf(D_LLj))
        # D_LLj[i] = D_LL_{i+1/2}
        # taui = np.array([tau(L, Kpt) for L in Li], dtype=float)
        def upsilon(L, Kpt):
                return 1 / t
        upsiloni = np.array([upsilon(L, Kpt) for L in Li], dtype=float)
        Xi = np.array([-Dt*Li[i+1]**2*D_LLj[i]/(DL**2*Lj[i]**2)
                       for i in range(nL-2)], dtype=float)
        # Yi = np.array([1+Dt/taui[i]+(Dt*Li[i+1]**2/DL**2)*
        #                (D_LLj[i]/Lj[i]**2+D_LLj[i+1]/Lj[i+1]**2)
        #                for i in range(nL-1)], dtype=float)
        Yi = np.array([1+Dt*upsiloni[i]+(Dt*Li[i+1]**2/DL**2)
                       *(D_LLj[i]/Lj[i]**2+D_LLj[i+1]/Lj[i+1]**2)
                       for i in range(nL-2)], dtype=float)
        Zi = np.array([-Dt*Li[i+1]**2*D_LLj[i+1]/(DL**2*Lj[i+1]**2)
                       for i in range(nL-2)], dtype=float)
        Ab = np.array([np.concatenate((np.zeros(2, dtype=float), Zi)),
                       np.concatenate((np.zeros(1, dtype=float), Yi,
                                       np.zeros(1, dtype=float))),
                       np.concatenate((Xi, np.zeros(2, dtype=float)))],
                      dtype=float)
        Ab = Ab[:, 1:-1]
        Ab[0, 0] = 0
        Ab[-1, -1] = 0
        # To be used in solve_banded
        y = initial.copy()
        y[0] -= Xi[0] * uLt[i]
        y[-1] -= Zi[-1] * uRt[i]
        ut = scipy.linalg.solve_banded((1, 1), Ab, y)
        ui[:, i] = ut
        initial = ut

    U_initial = np.zeros((nL), dtype=float)
    U_initial[0] = uLt[0]
    U_initial[1 : -1] = ui[:, 0]
    U_initial[-1] = uRt[0]
    U_final = np.zeros((nL, nT), dtype=float)
    U_final[0, :] = uLt
    U_final[1 : -1, :] = ui
    U_final[-1, :] = uRt
    return (Li, U_final.transpose())

def prediction_phase(orb_t_range, Dt, orb_boundary_times, orb_left,
                     orb_right, nRuns, perturbed_log_initial,
                     perturbed_Kp_data, L_range, nL, tau):
    nT = round((orb_t_range[1] - orb_t_range[0]) / Dt) + 1

    def uL(t):
        return np.exp(interpolate1D(orb_boundary_times, orb_left, t))

    def uR(t):
        return np.exp(interpolate1D(orb_boundary_times, orb_right, t))

    def D_LL(L, Kpt):
        return (10 ** (0.506 * Kpt - 9.325)) * L ** (10)

    # model_PSDs = []
    model_log_PSDs = []
    for j in range(nRuns):
        run_log_initial = perturbed_log_initial[j]
        Kp_data = perturbed_Kp_data[j]
        model_PSD = solve_diffusion(L_range, orb_t_range, nL, nT,
                                    np.exp(run_log_initial), D_LL, tau,
                                    uL, uR, Kp_data)[1]
        model_log_PSDs.append(np.log(model_PSD))
    finals = [PSD[-1, :] for PSD in model_log_PSDs]
    return {
        "model_log_PSDs" : model_log_PSDs,
        "finals" : finals,
        "nT" : nT
    }

def analysis_phase(VAP_log, ts, orb_t_range, model_log_PSDs, finals, nRuns,
                   H_all_data, nT):
    P_f = np.cov(np.transpose(finals))
    filtered_VAP_log = []
    filtered_ts = []
    H = []
    for i, log in enumerate(VAP_log):
        if not np.isnan(log):
            filtered_VAP_log.append(log)
            filtered_ts.append(ts[i])
            H.append(H_all_data[i])

    if H != []:
        filtered_VAP_log = np.array(filtered_VAP_log, dtype=float)
        H = np.array(H, dtype=float)  # projecting from model space to
        # observation space
        Ht = H.transpose()
        H_inv = (np.array(H > 0, dtype=int)).transpose()
        # maps from observation space to model space such that
        # H @ H_inv = I
        proj_times = H_inv @ filtered_ts  # the times of out VAP_points
        # projected onto the model space
        model_times = np.linspace(orb_t_range[0], orb_t_range[1], nT)
        model_point_logs = [[interpolate1D(model_times,
                                           model_log[:, i], t)
                             if t != 0 else 0
                             for i, t in enumerate(proj_times)]
                            for model_log in model_log_PSDs]
        average_final = np.average(finals, axis=0)
        R = np.diag(H @ average_final) / 2
        K = P_f @ Ht @ np.linalg.inv(H @ P_f @ Ht + R)
        innovations = [filtered_VAP_log - H @ model_point_log
                       for model_point_log in model_point_logs]
        perturbed_log_initial = [finals[i] + K @ innovations[i]
                                 for i in range(nRuns)]
    return {
        "innovations" : innovations,
        "perturbed_log_initial": perturbed_log_initial
    }

def kalman(L_range, t_range, orbits, orbit_boundary_times, orbit_log_lefts,
           orbit_log_rights, nRuns, log_initial, Kp_data, VAP_times, VAP_logs,
           H_all_data):
    """
    PDE is $\frac{dF}{dt}
    =L^2\frac{d}{dL}\left(\frac{1}{L^2}D_{LL}\frac{dF}{dL}\right)
    -\frac{F}{tau(L)}$
    L_range is the range of L that we are considering (a tuple)
    t_range is the range of t that we are considering (a tuple)
    initial condition $F(x, t_{min})=f0(x)$
    boundary conditions are $F(x_{min}, t)=F(x_{min}, t_{min})$;
                            $F(x_{max}, t)=F(x_{max}, t_{min})$
    """
    Kp_times = sorted(Kp_data.keys())
    t_range_Kp = (Kp_times[0], Kp_times[-1])
    assert t_range_Kp[0] <= t_range[0] and t_range_Kp[1] >= t_range[1]
    perturbed_log_initial = [np.array([f * random.gauss(1, 0.3)
                                       for f in log_initial], dtype=float)
                             for i in range(nRuns)]
    perturbed_Kp_data = [perturb_Kp_data(Kp_data) for i in range(nRuns)]
    DL = 0.01
    # Dt = datetime.timedelta(minutes=15)
    Dt = 0.01
    nL = round((L_range[1]-L_range[0]) / DL) + 1

    def tau(L, Kpt):
        """
        The function to be passed into the model's loss term. Taken from
        Shprits et al. (2005) as 3/Kp, and if Kp is zero, this function returns
        "infinity".
        """
        if Kpt != 0:
            return 3 / Kpt
        else:
            return np.inf

    average_model_logs = []
    average_innovations = []

    for o in range(len(orbits)):
        print("o", o)
        orb_t_range = orbits[o]
        orb_boundary_times = orbit_boundary_times[o]
        orb_left = orbit_log_lefts[o]
        orb_right = orbit_log_rights[o]
        predicted = prediction_phase(orb_t_range, Dt, orb_boundary_times,
                                     orb_left, orb_right, nRuns,
                                     perturbed_log_initial, perturbed_Kp_data,
                                     L_range, nL, tau)
        model_log_PSDs = predicted["model_log_PSDs"]
        finals = predicted["finals"]
        nT = predicted["nT"]
        ts = VAP_times[o]
        VAP_log = VAP_logs[o]
        if np.all(np.isnan(VAP_log)):
            perturbed_log_initial = finals
        else:
            analysed = analysis_phase(VAP_log, ts, orb_t_range, model_log_PSDs,
                                      finals, nRuns, H_all_data, nT)
            innovations = analysed["innovations"]
            perturbed_log_initial = analysed["perturbed_log_initial"]
            average_model_log = np.average(model_log_PSDs, axis=0)
            average_model_logs.append(average_model_log)
            average_innovation = np.average(innovations, axis=0)
            innovation_iterator = (i for i in average_innovation)
            average_innovation = []
            for t in ts:
                if np.isnan(t):
                    average_innovation.append(np.nan)
                else:
                    average_innovation.append(next(innovation_iterator))
            average_innovations.append(average_innovation)
    average_model_logs = np.concatenate(average_model_logs, axis=0)
    average_innovations = np.array(average_innovations)
    return average_innovations, average_model_logs


def interpolate1D(xi, fi, x):
    """
    Takes two series xi and fi of the same length, and returns the f value
    corresponding to finding the two values of xi nearest to x, and linearly
    interpolating between the corresponding y-values.
    """
    assert len(xi) == len(fi)
    if x < xi[0]:
        raise IndexError("Data not available at this point!")
    for i, p in enumerate(xi[1:]):
        if x <= p:
            p0 = xi[i]
            k = (x-p0)/(p-p0)
            return (1-k)*fi[i]+k*fi[i+1]
    if x == xi[-1]:
        return fi[-1]
    else:
        raise IndexError("Data not available at this point!")

def interpolate2D(xi, yi, fi, x, y):
    """
    Takes two lists xi and yi of length nx and ny, each of which is
    equally spaced (like np.linspace), and a third array fi of shape (n, n).
    It then finds f corresponding to linearly interpolating in x and y,
    and returns it.
    """
    nx, ny = len(xi), len(yi)
    assert(fi.shape == (nx, ny))
    mx = (x-xi[0])/(xi[-1]-xi[0])
    px = math.floor(mx*(nx-1))
    rx = mx*(nx-1) - px
    my = (y - yi[0]) / (yi[-1] - yi[0])
    py = math.floor(my*(ny-1))
    ry = my*(ny-1) - py
    if px == nx-1:
        assert(x == xi[-1])
        px1 = px
    else:
        px1 = px + 1
    if py == ny-1:
        assert (y == yi[-1])
        py1 = py
    else:
        py1 = py + 1
    assert(0 <= px <= px1 <= nx-1)
    assert(0 <= py <= py1 <= ny-1)
    assert(0 <= rx <= 1)
    assert(0 <= ry <= 1)
    f = ((1-rx) * (1-ry) * fi[px, py] + (1-rx) * ry * fi[px, py1]
         + rx * (1-ry) * fi[px1, py] + rx * ry * fi[px1, py1])
    return f

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
    # for dir_path in ["day", "week", "month", "month2"]:
    # for dir_path in ["day"]:
    for dir_path in ["week"]:
        Kp_data = read_Kp(dir_path + "/Kp_data.lst")
        Kp_data = {timeToDays(t): Kp for t, Kp in Kp_data.items()}
        with open(dir_path + '/kalman_data.json', "r") as inputJSON:
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

        all_boundary_times = []
        all_lefts = []
        all_rights = []
        for i, orbit in enumerate(orbit_boundary_times):
            all_boundary_times += list(orbit)
            all_lefts += list(orbit_log_lefts[i])
            all_rights += list(orbit_log_rights[i])
        boundaries = []
        for i, t in enumerate(all_boundary_times):
            boundaries.append((t, all_lefts[i], all_rights[i]))
        boundaries = sorted(set(boundaries))
        all_boundary_times = []
        all_lefts = []
        all_rights = []
        for t, l, r in boundaries:
            all_boundary_times.append(t)
            all_lefts.append(l)
            all_rights.append(r)

        DL = 0.01
        nL = round((L_range[1] - L_range[0]) / DL) + 1
        DT =  0.01
        nT = round((t_range[1] - t_range[0]) / DT) + 1
        def tau(L, Kpt):
            return 3 / Kpt if Kpt != 0 else np.inf

        def uL(t):
            return np.exp(interpolate1D(all_boundary_times, all_lefts, t))

        def uR(t):
            return np.exp(interpolate1D(all_boundary_times, all_rights, t))

        def D_LL(L, Kpt):
            return (10**(0.506*Kpt-9.325))*L**(10)

        Li, diffusion_output = solve_diffusion(L_range, t_range, nL, nT,
                                           np.exp(log_initial), D_LL, tau,
                                           uL, uR, Kp_data)


        kalman_stuff = kalman(L_range, t_range, orbits, orbit_boundary_times,
                              orbit_log_lefts, orbit_log_rights, 100,
                              log_initial, Kp_data, VAP_times, VAP_logs,
                              H_all_data)
        innovations, kalman_logs = kalman_stuff
        output_dict = {"diffusion_output" : diffusion_output.tolist(),
            "kalman_output" : kalman_logs.tolist(),
            "innovations" : innovations.tolist(),
        }
        with open(dir_path + '/kalman_output.json', "w") as outputJSON:
            kalman_data = json.dump(output_dict, outputJSON)
        print(diffusion_output.shape)
        print(kalman_logs.shape)
        print(innovations)
        print(np.nanmean(innovations, axis=0))
        print(np.nanmean(innovations))

