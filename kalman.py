import random

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import math
import datetime
from read_Kp import read_Kp
from spacepy import pycdf
from matplotlib.colors import LogNorm


def solve_diffusion(LRange, tRange, nL, nT, f0, D_LL, tau, uL, uR, Kp_data):
    """
    PDE is $\frac{dF}{dt}
    =L^2\frac{d}{dL}\left(\frac{1}{L^2}D_{LL}\frac{dF}{dL}\right)
    -\frac{F}{tau(L)}$
    LRange is the range of L that we are considering (a tuple)
    tRange is the range of t that we are considering (a tuple)
    initial condition $F(x, t_{min})=f0(x)$
    boundary conditions are $F(x_{min}, t)=F(x_{min}, t_{min})$;
                            $F(x_{max}, t)=F(x_{max}, t_{min})$
    """
    Li = np.linspace(LRange[0], LRange[1], num=nL)
    times = np.linspace(tRange[0], tRange[1], num=nT)
    DL = (LRange[1] - LRange[0])/(nL-1)
    Dt = (tRange[1] - tRange[0])/(nT-1)
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
        # D_LLj[i] = D_LL_{i+1/2}
        # taui = np.array([tau(L, Kpt) for L in Li], dtype=float)
        def upsilon(L, Kpt):
            t = tau(L, Kpt)
            if t == "infinity":
                return 0
            else:
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
        L = initial.copy()
        L[0] -= Xi[0] * uLt[i]
        L[-1] -= Zi[-1] * uRt[i]
        ut = scipy.linalg.solve_banded((1, 1), Ab, L)
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
    return (Li, U_final)

def kalman(LRange, tRange, nL, nT, f0, D_LL, tau, uL, uR, Kp_data, ):
    """
    PDE is $\frac{dF}{dt}
    =L^2\frac{d}{dL}\left(\frac{1}{L^2}D_{LL}\frac{dF}{dL}\right)
    -\frac{F}{tau(L)}$
    LRange is the range of L that we are considering (a tuple)
    tRange is the range of t that we are considering (a tuple)
    initial condition $F(x, t_{min})=f0(x)$
    boundary conditions are $F(x_{min}, t)=F(x_{min}, t_{min})$;
                            $F(x_{max}, t)=F(x_{max}, t_{min})$
    """
    orbits =

    return (Li, density)

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

def timeToDays(t, jan1):
    """
    Takes a time (t) in the form of datetime.datetime and returns the (float)
    number of days + 1 since midnight on 1st January of that year (jan1,
    provided as a datetime.datetime.
    """
    delta = t-jan1
    return delta.total_seconds()/86400 + 1

if __name__ == "__main__":
    # for dir in ["day", "week", "month", "month2"]:
    for dir in ["day"]:
        # dir = "day"
        # dir = "week"
        # dir = "month"
        # dir = "month2"
        Kp_data = read_Kp(dir + "/Kp_data.lst")
        t0 = min(Kp_data.keys())
        tf = max(Kp_data.keys())
        tRangeKp = (t0/24 + 1, tf/24 + 1) # in days

        def perturbKp(Kp_data, perturbation=0.25):
            print(1)
            i=0
            perturbed = {}
            for t, data in Kp_data.items():
                print(i)
                perturbed[t] = data * random.gauss(1, perturbation)
                i += 1
                print(len(Kp_data), i)
            print(1)
            return perturbed

        def Kp(Kp_data, t):
            """
            t in days
            """
            hour = (t - 1) * 24
            if t == tRangeKp[1]:
                return Kp_data[round(hour)]
            if hour < t0 or hour > tf:
                raise IndexError("Data not available for that time!")
            t1 = math.floor(hour)
            t2 = t1+1
            d = hour - t1
            return Kp_data[t1]*(1-d) + Kp_data[t2]*d


        def tau(L, Kpt):
            """
            The function to be passed into the model's loss term. Taken from
            Shprits et al. (2005) as 3/Kp, and if Kp is zero, this function returns
            "infinity".
            """
            if Kpt != 0:
                return 3 / Kpt
            else:
                return "infinity"

        cdf = pycdf.CDF(dir + "/kalman_input.cdf").copy()
        cdfTimes = cdf["times"]
        jan1 = datetime.datetime(cdfTimes[0].year, 1, 1)
        LRange = (3.1, 5.3)
        cdfTimes = [timeToDays(t, jan1) for t in cdfTimes]
        tRangeCDF = (cdfTimes[0], cdfTimes[-1])
        tRange = (max(tRangeKp[0], tRangeCDF[0]), min(tRangeKp[1], tRangeCDF[1]))
        assert(tRange[0] <= tRange[1])
        print(tRangeKp, tRangeCDF)
        DL = 0.01
        Dt = 0.01
        epochLength = 0.1
        nEpochs = int((LRange[-1] - LRange[0])/epochLength) + 1
        timesPerEpoch = int(epochLength/Dt) + 1
        nL = int((LRange[-1] - LRange[0])/DL) + 1
        cdfLi = cdf["Li"]
        PSD = cdf["PSD"]

        def f0(L):
            return np.exp(interpolate1D(cdfLi, np.log(PSD[0, :]), L))

        def uL(t):
            return np.exp(interpolate2D(cdfTimes, cdfLi, np.log(PSD), t,
                                        LRange[0]))

        def uR(t):
            return np.exp(interpolate2D(cdfTimes, cdfLi, np.log(PSD), t,
                                        LRange[1]))

        def D_LL(L, Kpt):
            return (10**(0.506*Kpt-9.325))*L**(10)

        modelLi, modelPSD = kalman(LRange, tRange, nL, timesPerEpoch, f0, D_LL, tau, uL,
                                   uR, Kp_data, 4, 0.25, nEpochs)
        modelTimes = np.linspace(tRange[0], tRange[1], nEpochs * timesPerEpoch)

        fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(10,15))
        fig.suptitle("Model performance vs. VAP data", fontsize=16)
        X = [cdfTimes for i in range(len(cdfLi))]
        Y = np.transpose([cdfLi for i in range(len(cdfTimes))])
        Z = PSD.transpose()
        c = ax0.pcolor(X, Y, Z, norm=LogNorm(vmin=1e-9, vmax=1e-4),
                       cmap=plt.cm.rainbow)

        fig.colorbar(c, ax=ax0)
        ax0.set_title("VAP Phase Space Density")
        ax0.set_xlabel("Time (days)")
        ax0.set_ylabel('$L (R_E)$')

        X = [modelTimes for i in range(len(modelLi))]
        Y = np.transpose([modelLi for i in range(len(modelTimes))])
        Z = modelPSD
        print(nEpochs, timesPerEpoch)
        c = ax1.pcolor(X, Y, Z, norm=LogNorm(vmin=1e-9, vmax=1e-4),
                       cmap=plt.cm.rainbow)
        fig.colorbar(c, ax=ax1)
        ax1.set_title("Model Phase Space Density")
        ax1.set_xlabel('Time (days)')
        ax1.set_ylabel('$L (R_E)$')

        Kp_times = [t for t in Kp_data.keys() if tRange[0] <= t/24+1 <= tRange[1]]
        Kp_data = [Kp_data[t] for t in Kp_times]
        # Kp_data = [perturbed[t] for t in Kp_times]
        assert(Kp_times != [])
        ax2.set_title("Kp data")
        ax2.plot([t/24+1 for t in Kp_times], Kp_data)
        ax2.set_xlabel('Time (days)')
        ax2.set_ylabel('Kp')
        ax2.set(xlim=(tRange[0], tRange[1]))
        plt.tight_layout()
        plt.savefig(dir + '/output.png')
        plt.show()
        print("Done!")

        # n = 7
        # for i in range(n):
        #     j = ((nT-1) * i)//(n-1)
        #     t = modelTimes[j]
        #     predicted = modelPSD[:, j]
        #     plt.plot(modelLi, predicted, "r", label = "Predicted")
        #     actual = [interpolate1D(cdfTimes, PSD[:, k], t)
        #               for k in range(len(PSD[0, :]))]
        #     plt.plot(cdfLi, actual, "b", label="Actual")
        #     plt.yscale('log')
        #     plt.ylim(1e-10, 1e-4)
        #     plt.xlabel("L value")
        #     plt.ylabel("Phase Space Density")
        #     plt.title("Phase Space Density at t=" + str(t)[:7] + " days")
        #     plt.legend()
        #     plt.show()

        # for i in range(nL):
        #     if i % 100 == 0:
        #         L=np.linspace(LRange[0], LRange[1], nL)[i]
        #         predicted = Ui[i, :]
        #         times = np.linspace(tRange[0], tRange[1], nT)
        #         plt.plot(times, predicted, "r",label = "Predicted")
        #         actual = [interpolate1D(cdfLi, PSD[j, :], L)
        #                   for j in range(len(PSD[:, 0]))]
        #         print(len(cdfTimes), PSD.shape)
        #         plt.plot(cdfTimes, actual, "b", label="Actual")
        #         plt.yscale('log')
        #         plt.ylim(1e-10, 1e-4)
        #         plt.xlabel("L value")
        #         plt.ylabel("Phase Space Density")
        #         plt.title("Phase Space Density at L=" + str(L)[:7])
        #         plt.legend()
        #         plt.show()


