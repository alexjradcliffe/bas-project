import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import math
import datetime
from read_Kp import read_Kp
from spacepy import pycdf

def solve_diffusion(LRange, tRange, nL, nT, f0, D_LL, tau, uL, uR, Kp):
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
    Li = np.linspace(LRange[0], LRange[1], num=nL+1)
    times = np.linspace(tRange[0], tRange[1], num=nL+1)
    DL = (LRange[1] - LRange[0])/(nL)
    Dt = (tRange[1] - tRange[0])/(nT)
    initial = np.array([f0(L) for L in Li[1 : -1]], dtype=float)
    uLt = [uL(t) for t in times] # left boundary condition
    uRt = [uR(t) for t in times] # right boundary condition
    Lj = np.array([(Li[i]+Li[i+1])/2 for i in range(nL)], dtype=float)
    # Lj[i] = L_{i+1/2}
    ui = np.zeros((nL - 1, nT+1), dtype=float) #excludes x=0 and x=1
    ui[:, 0] = initial
    for i in range(1, nT + 1):
        t = tRange[0] + i * Dt
        Kpt = Kp(t)
        D_LLj = np.array([(D_LL(Li[i], Kpt) + D_LL(Li[i + 1], Kpt)) / 2 for i in range(nL)], dtype=float)
        # D_LLj[i] = D_LL_{i+1/2}
        # taui = np.array([tau(L, Kpt) for L in Li], dtype=float)
        def upsilon(L, Kpt):
            t = tau(L, Kpt)
            if t == "infinity":
                return 0
            else:
                return 1 / t
        upsiloni = np.array([upsilon(L, Kpt) for L in Li], dtype=float)
        Xi = np.array([-Dt*Li[i+1]**2*D_LLj[i]/(DL**2*Lj[i]**2) for i in range(nL-1)], dtype=float)
        # Yi = np.array([1+Dt/taui[i]+(Dt*Li[i+1]**2/DL**2)*(D_LLj[i]/Lj[i]**2+D_LLj[i+1]/Lj[i+1]**2) for i in range(nL-1)], dtype=float)
        Yi = np.array([1+Dt*upsiloni[i]+(Dt*Li[i+1]**2/DL**2)*(D_LLj[i]/Lj[i]**2+D_LLj[i+1]/Lj[i+1]**2) for i in range(nL-1)], dtype=float)
        Zi = np.array([-Dt*Li[i+1]**2*D_LLj[i+1]/(DL**2*Lj[i+1]**2) for i in range(nL-1)], dtype=float)
        Ab = np.array([np.concatenate((np.zeros(2, dtype=float), Zi)),
                       np.concatenate((np.zeros(1, dtype=float), Yi, np.zeros(1, dtype=float))),
                       np.concatenate((Xi, np.zeros(2, dtype=float)))], dtype=float)
        Ab = Ab[:, 1:-1]
        Ab[0, 0] = 0
        Ab[-1, -1] = 0
        # To be used in solve_banded
        L = initial
        L[0] -= Xi[0] * uLt[i]
        L[-1] -= Zi[-1] * uRt[i]
        ut = scipy.linalg.solve_banded((1, 1), Ab, L)
        ui[:, i] = ut
        initial = ut

    U_initial = np.zeros((nL+1), dtype=float)
    U_initial[0] = uLt[0]
    U_initial[1 : -1] = ui[:, 0]
    U_initial[-1] = uRt[0]
    U_final = np.zeros((nL+1, nT+1), dtype=float)
    U_final[0, :] = uLt
    U_final[1 : -1, :] = ui
    U_final[-1, :] = uRt
    return (Li, U_final)

def interpolate1D(xi, yi, x):
    """
    Takes two series xi and yi of the same length, and returns the y value
    corresponding to finding the two values of xi nearest to x, and linearly
    interpolating between the corresponding y-values.
    """
    assert len(xi) == len(yi)
    if x < xi[0]:
        raise IndexError("Data not available at this point!")
    for i, p in enumerate(xi[1:]):
        if x <= p:
            p0 = xi[i]
            k = (x-p0)/(p-p0)
            return (1-k)*yi[i]+k*yi[i+1]
    if x == xi[-1]:
        return yi[-1]
    else:
        raise IndexError("Data not available at this point!")

def timeToDays(t, jan1):
    """
    Takes a time (t) in the form of datetime.datetime and returns the (float)
    number of days + 1 since midnight on 1st January of that year (jan1,
    provided as a datetime.datetime.
    """
    delta = t-jan1
    return delta.total_seconds()/86400 + 1

if __name__ == "__main__":
    Kp_data = read_Kp("20170908/Kp_data.lst")
    t0 = min(Kp_data.keys())
    tf = max(Kp_data.keys())
    # t0, tf, Kp_data = read_Kp("20171226/Kp_data.lst")
    Kp_range = (t0, tf)

    def Kp(t):
        """
        t in days
        """
        hour = (t - 1) * 24
        if hour < t0 or hour > tf:
            raise IndexError("Data not available for that time!")
        if t == tf:
            return Kp_data[t]
        t1 = math.floor(hour)
        d = hour - t1
        return Kp_data[t1]*(1-d) + Kp_data[t1]*d


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

    cdf = pycdf.CDF("20170908/output.cdf").copy()
    # cdf = pycdf.CDF("20171226/output.cdf").copy()
    cdfTimes = cdf["times"]
    jan1 = datetime.datetime(cdfTimes[0].year, 1, 1)
    LRange = (3.1, 5)
    nL = 1000
    nT = 1000
    cdfTimes = [timeToDays(t, jan1) for t in cdfTimes]
    tRange = (cdfTimes[0], cdfTimes[-1])
    print(tRange)
    Li = cdf["Li"]
    PSD = cdf["PSD"]
    cdfLi = Li


    def f0(L):
        return interpolate1D(Li, PSD[0, :], L)

    def uL(t):
        return interpolate1D(cdfTimes, PSD[:, 0], t)

    def uR(t):
        return interpolate1D(cdfTimes, PSD[:, -1], t)

    def D_LL(L, Kpt):
        return (10**(0.506*Kpt-9.325))*L**(10)

    Li, Ui = solve_diffusion(LRange, tRange, nL, nT, f0, D_LL, tau, uL, uR, Kp)

    # for i in range(nT+1):
    #     if i % 100 == 0:
    #         t=np.linspace(tRange[0], tRange[1], nT+1)[i]
    #         predicted = Ui[:, i]
    #         plt.plot(Li, predicted, "r", label = "Predicted")
    #         actual = [interpolate1D(cdfTimes, PSD[:, j], t) for j in range(len(PSD[:, 0]))]
    #         plt.plot(cdfLi, actual, "b", label="Actual")
    #         plt.yscale('log')
    #         plt.ylim(1e-10, 1e-4)
    #         plt.xlabel("L value")
    #         plt.ylabel("Phase Space Density")
    #         plt.title("Phase Space Density at t=" + str(t)[:7])
    #         plt.legend()
    #         plt.show()

    for i in range(nL+1):
        if i % 100 == 0:
            L=np.linspace(LRange[0], LRange[1], nL+1)[i]
            predicted = Ui[i, :]
            plt.plot(np.linspace(tRange[0], tRange[1], nL+1), predicted, "r", label = "Predicted")
            actual = [interpolate1D(cdfLi, PSD[j, :], L) for j in range(len(PSD[0, :]))]
            plt.plot(cdfTimes, actual, "b", label="Actual")
            plt.yscale('log')
            plt.ylim(1e-10, 1e-4)
            plt.xlabel("L value")
            plt.ylabel("Phase Space Density")
            plt.title("Phase Space Density at L=" + str(L)[:7])
            plt.legend()
            plt.show()


