import matplotlib
matplotlib.use("TKAgg")
from matplotlib import pyplot as plt
import numpy as np 
import sksundae as sun

"""Robertson chemical kinetics ODE system.
This is a classic stiff ODE problem defined by the reactions:
    A --> B  (rate constant k1 = 0.04)
    B + B --> B + C  (rate constant k2 = 1e4)
    B + C --> A + C + C  (rate constant k3 = 3e7)
The system of ODEs is given by:
    dy0/dt = -k1*y0 + k3*y1*y2
    dy1/dt = k1*y0 - k2*y1^2 - k3*y1*y2
    dy2/dt = k2*y1^2
with initial conditions y0(0) = 1, y1(0) = 0, y2(0) = 0.

The main issue with this problem is its stiffness. For numerical modeling purposes, using CVODE does not work with DAEs. 
Instead, we use IDA solver from Sundials, which is designed for differential-algebraic equations (DAEs).
"""

def resfn(t, y, yp, res):
    res[0] = yp[0] + 0.04*y[0] - 1e4*y[1]*y[2]
    res[1] = yp[1] - 0.04*y[0] + 1e4*y[1]*y[2] + 3e7*y[1]**2
    res[2] = y[0] + y[1] + y[2] - 1

tspan = np.logspace(-6, 6, 50)
y0 = np.array([1, 0, 0])
yp0 = np.array([-0.04, 0.04, 0])
    
solver = sun.ida.IDA(resfn, atol=1e-8, algebraic_idx=[2])
soln = solver.solve(tspan, y0, yp0)
print(soln)

soln.y[:,1] *= 1e4  # scale the y1 values for plotting

plt.semilogx(soln.t, soln.y)
plt.legend(['y0', 'y1', 'y2'])
plt.xlabel(r"$t$");
plt.ylabel("concentration");
plt.show()