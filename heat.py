import numpy as np
import matplotlib.pyplot as plt
import scipy

nX = 100
nT = nX

# PDE is $d^2u/dx^2=du/dt$
# x-range is (0, 1) — t-range is (0, 1)
# initial condition $u(x, 0)=\sin(2\pi x)$
# boundary conditions are $u(0, t)=u(1, t)=0$

xi = np.linspace(0, 1, num=nX)
ti = np.linspace(0, 1, num=nT)
Dx = 1/(nX - 1)
Dt = 1/(nT - 1)
initial = np.sin(np.pi*xi[1 : -1])
ui = np.zeros((nX - 2, nT)) #excludes x=0 and x=1
ui[:, 0] = initial
# If our equation is Au=x
k = Dt / Dx ** 2
# A = 2*np.identity(nX - 2)
# for j in range(nX - 3):
#     A[j, j + 1] = -1
#     A[j + 1, j] = -1
# A *= k
# A += np.identity(nX - 2)
Ab = np.array([[-k], [1+2*k], [-k]])
Ab = Ab @ np.ones(nX-2)
Ab[0, 0] = 0
Ab[-1, -1] = 0

for i in range(1, nT):
    x = initial
    x[0] += 0
    x[- 1] += 0
    ut = np.linalg.inv(A) @ x
    # ut = scipy.linalg.solve_banded((1, 1), Ab, x)
    ui[:, i] = ut
    initial = ut

U_predicted = np.zeros((nX, nT))
U_predicted[1 : -1,:] = ui

U_actual = np.array([[(np.sin(np.pi*x))*np.exp(-t*np.pi**2) for t in ti] for x in xi])
# print(U_predicted[:, 1])
# print(U_actual[:, 1])
ratio = U_predicted[1:-1, :]/U_actual[1:-1, :]
plt.plot(xi, U_actual[:, 50], "r")
plt.plot(xi, U_predicted[:, 50], "b")
plt.show()