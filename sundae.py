import matplotlib
matplotlib.use("TKAgg")
from matplotlib import pyplot as plt
import numpy as np 
import sksundae as sun

def rhsfn(t, y, yp):
    yp[0] = y[1]
    yp[1] = 1000*(1 - y[0]**2)*y[1] - y[0]

tspan = np.array([0, 3000])
y0 = np.array([2, 0])

solver = sun.cvode.CVODE(rhsfn)
sln = solver.solve(tspan, y0)
print(sln)

print(sln.t)
print(sln.y[:,0])

plt.plot(sln.t, sln.y[:, 0], label='y0 (position)')
plt.xlabel('Time')
plt.ylabel('Solution')
plt.title('Van der Pol Oscillator Solution using CVODE')
plt.legend()
plt.grid()
plt.show()