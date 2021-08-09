import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

def solve_diffusion(LRange, tRange, nL, nT, f0, D_LL, tau):
    """
    PDE is $\frac{dF}{dt}
    =L^2\frac{d}{dL}\left(\frac{1}{L^2}D_{LL}\frac{dF}{dL}\right)
    -\frac{F}{tau(L)}$
    LRange is the range of L that we are considering (a tuple)
    tRange is the range of t that we are considering (a tuple)
    initial condition $F(x, t_{min})=f0(x)$
    boundary conditions are $F(x_{min}, t)=F(x_{min}, t_{min}),$
                            $F(x_{max}, t)=F(x_{max}, t_{min})$
    """
    Li = np.linspace(LRange[0], LRange[1], num=nL+1)
    DL = (LRange[1] - LRange[0])/(nL)
    Dt = (tRange[1] - tRange[0])/(nT)
    initial = np.array([f0(L) for L in Li[1 : -1]], dtype=float)
    uL = f0(LRange[0])*np.ones(nT+1, dtype=float) # left boundary condition
    uR = f0(LRange[1])*np.ones(nT+1, dtype=float) # right boundary condition
    D_LLj = np.array([(D_LL(Li[i])+D_LL(Li[i+1]))/2 for i in range(nL)], dtype=float)
    Lj = np.array([(Li[i]+Li[i+1])/2 for i in range(nL)], dtype=float)
    # D_LLj[i] = D_LL_{i+1/2}
    # Lj[i] = L_{i+1/2}
    taui = np.array([tau(L) for L in Li], dtype=float)
    ui = np.zeros((nL - 1, nT+1), dtype=float) #excludes x=0 and x=1
    ui[:, 0] = initial
    Xi = np.array([-Dt*Li[i+1]**2*D_LLj[i]/(DL**2*Lj[i]**2) for i in range(nL-1)], dtype=float)
    Yi = np.array([1+Dt/taui[i]+(Dt*Li[i+1]**2/DL**2)*(D_LLj[i]/Lj[i]**2+D_LLj[i+1]/Lj[i+1]**2) for i in range(nL-1)], dtype=float)
    Zi = np.array([-Dt*Li[i+1]**2*D_LLj[i+1]/(DL**2*Lj[i+1]**2) for i in range(nL-1)], dtype=float)
    Ab = np.array([np.concatenate((np.zeros(2, dtype=float), Zi)),
                   np.concatenate((np.zeros(1, dtype=float), Yi, np.zeros(1, dtype=float))),
                   np.concatenate((Xi, np.zeros(2, dtype=float)))], dtype=float)
    Ab = Ab[:, 1:-1]
    Ab[0, 0] = 0
    Ab[-1, -1] = 0
    # To be used in solve_banded
    for i in range(1, nT+1):
        L = initial
        L[0] -= Xi[0] * uL[i]
        L[-1] -= Zi[-1] * uR[i]
        ut = scipy.linalg.solve_banded((1, 1), Ab, L)
        ui[:, i] = ut
        initial = ut

    U_initial = np.zeros((nL+1), dtype=float)
    U_initial[0] = uL[0]
    U_initial[1 : -1] = ui[:, 0]
    U_initial[-1] = uR[0]
    U_final = np.zeros((nL+1), dtype=float)
    U_final[0] = uL[-1]
    U_final[1 : -1] = ui[:, -1]
    U_final[-1] = uR[-1]
    return (Li, U_final)

LRange = (2, 8)
tRange = (0, 1000)
nL = 1000
nT = 10000

def tau(L):
    return 1

def f0(L):
    return 0 if L < 8 else 1

def D_LL(L, Kp):
    return (10**(0.506*Kp-9.325))*L**(10)

Li, U_final = solve_diffusion(LRange, tRange, nL, nT, f0, lambda L: D_LL(L, 2), tau)
plt.plot(Li, U_final, "r", label = "Kp=2")
Li, U_final = solve_diffusion(LRange, tRange, nL, nT, f0, lambda L: D_LL(L, 6), tau)
plt.plot(Li, U_final, "b", label = "Kp=6")
plt.yscale('log')
plt.ylim(0.01, 10)
plt.xlabel("L value")
plt.ylabel("Phase Space Density")
plt.title("τ=1")
plt.legend()
plt.show()
def tau(L):
    return 10
Li, U_final = solve_diffusion(LRange, tRange, nL, nT, f0, lambda L: D_LL(L, 2), tau)
plt.plot(Li, U_final, "r", label = "Kp=2")
Li, U_final = solve_diffusion(LRange, tRange, nL, nT, f0, lambda L: D_LL(L, 6), tau)
plt.plot(Li, U_final, "b", label = "Kp=6")
plt.yscale('log')
plt.ylim(0.01, 10)
plt.xlabel("L value")
plt.ylabel("Phase Space Density")
plt.title("τ=10")
plt.legend()
plt.show()
