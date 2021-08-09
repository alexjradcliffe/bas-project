import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

xRange = (-1, 1)
tRange = (0, 1)
nX = 1000
nT = 1000
omega = np.pi

# PDE is $d^2u/dx^2=du/dt$
# x-range is (0, 1) — t-range is (0, 1)
# initial condition $u(x, 0)=\sin(2\pi x)$
# boundary conditions are $u(0, t)=u(1, t)=0$

xi = np.linspace(xRange[0], xRange[1], num=nX)
ti = np.linspace(tRange[0], tRange[1], num=nT)
Dx = (xRange[1] - xRange[0])/(nX - 1)
Dt = (tRange[1] - tRange[0])/(nT - 1)
initial = np.sin(omega*xi[1 : -1])
ui = np.zeros((nX - 2, nT)) #excludes x=0 and x=1
ui[:, 0] = initial
# If our equation is Au=x
k = Dt / Dx ** 2
print(k)
# A = 2*np.identity(nX - 2)
# for j in range(nX - 3):
#     A[j, j + 1] = -1
#     A[j + 1, j] = -1
# A *= k
# A += np.identity(nX - 2)
Ab = np.array([[-k], [1+2*k], [-k]])
Ab = Ab @ np.ones((1, nX-2))
Ab[0, 0] = 0
Ab[-1, -1] = 0
print(Ab)

for i in range(1, nT):
    x = initial
    x[0] += k * 0
    x[- 1] += k * 0
    # ut = np.linalg.inv(A) @ x
    ut = scipy.linalg.solve_banded((1, 1), Ab, x)
    ui[:, i] = ut
    initial = ut

U_predicted = np.zeros((nX, nT))
U_predicted[1 : -1,:] = ui

U_actual = np.array([[(np.sin(omega*x))*np.exp(-t*omega**2) for t in ti] for x in xi])
# print(U_predicted[:, 1])
# print(U_actual[:, 1])
ratio = U_predicted[1:-1, :]/U_actual[1:-1, :]
plt.plot(xi, U_actual[:, -1], "r")
plt.plot(xi, U_predicted[:, -1], "b")
plt.show()