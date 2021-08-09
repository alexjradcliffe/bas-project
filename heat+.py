import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

def solve_pde(xRange, tRange, nX, nT, f0, mu, tau):
    """
    PDE is $\frac{du}{dt}
    =\frac{d}{dx}\left(\mu(x)\frac{du}{dx}\right)
    -\frac{u}{tau(x)}$
    $\mu(x)=\tau(x)=x$
    x-range is (1, 2) — t-range is (0, 1)
    initial condition $u(x, 0)=\log(x)$
    boundary conditions are $u(0, t)=0, u(0, t)=\log2$
    """
    xi = np.linspace(xRange[0], xRange[1], num=nX)
    Dx = (xRange[1] - xRange[0])/(nX - 1)
    Dt = (tRange[1] - tRange[0])/(nT - 1)
    initial = f0(xi[1 : -1])
    uL = f0(xRange[0])*np.ones(nT) # left boundary condition
    uR = f0(xRange[1])*np.ones(nT) # right boundary condition
    mui = np.array([(mu(xi[i])+mu(xi[i+1]))/2 for i in range(nX - 1)])
    # mui[i] = mui_{i+1/2}
    taui = np.array([tau(x) for x in xi])
    ui = np.zeros((nX - 2, nT)) #excludes x=0 and x=1
    ui[:, 0] = initial
    # If our equation is Au=x
    m = Dt / Dx ** 2
    # A = 2*np.identity(nX - 2)
    # for j in range(nX - 3):
    #     A[j, j + 1] = -1
    #     A[j + 1, j] = -1
    # A *= k
    # A += np.identity(nX - 2)
    Ab = np.array([[-m*mui[i], 1+1/taui[i]+m*(mui[i]+mui[i + 1]), -m*mui[i+1]] for i in range(nX-2)]).transpose()
    Ab[0, 0] = 0
    Ab[-1, -1] = 0
    # To be used in solve_banded

    for i in range(1, nT):
        x = initial
        x[0] += m * mui[0] * uL[i]
        x[-1] += m * mui[-1] * uR[i]
        ut = scipy.linalg.solve_banded((1, 1), Ab, x)
        ui[:, i] = ut
        initial = ut

    U_initial = np.zeros((nX))
    U_initial[0] = uL[0]
    U_initial[1 : -1] = ui[:, 0]
    U_initial[-1] = uR[0]
    U_final = np.zeros((nX))
    U_final[0] = uL[-1]
    U_final[1 : -1] = ui[:, -1]
    U_final[-1] = uR[-1]

    # U_actual = np.array([[(np.sin(omega*x))*np.exp(-t*omega**2) for t in ti] for x in xi])
    # print(U_predicted[:, 1])
    # print(U_actual[:, 1])
    # ratio = U_predicted[1:-1, :]/U_actual[1:-1, :]
    # plt.plot(xi, U_actual[:, -1], "r")
    plt.plot(xi, U_initial, "b")
    plt.plot(xi, U_final, "r")
    plt.show()

xRange = (1, 2)
tRange = (0, 1)
nX = 100
nT = 100
def mu(x):
    return x
def tau(x):
    return x
def f0(x):
    return np.log(x)

solve_pde(xRange, tRange, nX, nT, f0, mu, tau)