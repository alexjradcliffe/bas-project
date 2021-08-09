import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

def solve_diffusion(LRange, tRange, nL, nT, f0, D_LL, tau):
    """
    PDE is $\frac{du}{dt}
    =\frac{d}{dx}\left(\mu(x)\frac{du}{dx}\right)
    -\frac{u}{tau(x)}$
    $\mu(x)=\tau(x)=x$
    x-range is (1, 2) — t-range is (0, 1)
    initial condition $u(x, 0)=\log(x)$
    boundary conditions are $u(0, t)=0, u(0, t)=\log2$
    """
    Li = np.linspace(LRange[0], LRange[1], num=nL)
    DL = (LRange[1] - LRange[0])/(nL - 1)
    Dt = (tRange[1] - tRange[0])/(nT - 1)
    initial = f0(Li[1 : -1])
    uL = f0(LRange[0])*np.ones(nT) # left boundary condition
    uR = f0(LRange[1])*np.ones(nT) # right boundary condition
    # D_LLi = np.array([D_LL(Li[i])+D_LL(Li[i+1]) for i in range(nL - 1)])
    # D_LLi[i] = D_LL_{i+1/2}
    mui = np.array([D_LL(Li[i])+D_LL(Li[i+1])/(((Li[i]+Li[i+1])/2)**2) for i in range(nL - 1)])
    # mu = D_LL/L^2
    # mu_i[i] = mu_{i+1/2}
    taui = np.array([tau(L) for L in Li])
    ui = np.zeros((nL - 2, nT)) #excludes x=0 and x=1
    ui[:, 0] = initial
    # If our equation is Au=x
    m = Dt / (DL ** 2)
    # A = 2*np.identity(nX - 2)
    # for j in range(nX - 3):
    #     A[j, j + 1] = -1
    #     A[j + 1, j] = -1
    # A *= k
    # A += np.identity(nX - 2)
    Ab = np.array([[-Li[i]**2*m*mui[i],
                    1+1/taui[i]+Li[i]**2*m*(mui[i]+mui[i + 1]),
                    -Li[i]**2*m*mui[i+1]] for i in range(nL-2)]).transpose()
    Ab[0, 0] = 0
    Ab[-1, -1] = 0
    # To be used in solve_banded

    for i in range(1, nT):
        L = initial
        L[0] += Li[0]**2 * m * mui[0] * uL[i]
        L[-1] += Li[-1]**2 * m * mui[-1] * uR[i]
        ut = scipy.linalg.solve_banded((1, 1), Ab, L)
        ui[:, i] = ut
        initial = ut

    U_initial = np.zeros((nL))
    U_initial[0] = uL[0]
    U_initial[1 : -1] = ui[:, 0]
    U_initial[-1] = uR[0]
    U_final = np.zeros((nL))
    U_final[0] = uL[-1]
    U_final[1 : -1] = ui[:, -1]
    U_final[-1] = uR[-1]

    # U_actual = np.array([[(np.sin(omega*x))*np.exp(-t*omega**2) for t in ti] for x in xi])
    # print(U_predicted[:, 1])
    # print(U_actual[:, 1])
    # ratio = U_predicted[1:-1, :]/U_actual[1:-1, :]
    # plt.plot(xi, U_actual[:, -1], "r")
    plt.plot(Li, U_initial, "b")
    plt.plot(Li, U_final, "r")
    plt.show()

LRange = (1, 2)
tRange = (0, 1)
nL = 100
nT = 100
def D_LL(L):
    return L
def tau(L):
    return L
f0 = np.log

solve_diffusion(LRange, tRange, nL, nT, f0, D_LL, tau)