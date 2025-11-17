"""
Kraehnert chemical kinetics for ammonia oxidation over Pt, which is a DAE system.
This is given by a set of elementary steps and one non elementary step (R5 is not elementary)
This mechanism has the particularity of having two independent active sites a and b, each requiring a site balance.
Requirements:
    numpy>=1.21.0
    scikits.odes>=2.6.3 (compiled from source code)
    matplotlib>=3.5.0

The variables are:
    -----FIXED parameters-----
    xNH3 (partial pressure of NH3)
    xO2 (partial pressure of O2)
    xNO (partial pressure of NO)
    P: total pressure in kPa
    T: temperature in K
    --------DAE IDA Variables--------
    y0: Theta_b (free site b)
    y1: Theta_b-NH3 (Adsorbed NH3 on site b)
    y2: Theta_a (free site a)
    y3: Theta_a-O (Adsorbed O on site a)
    y4: Theta_a-NO (Adsorbed NO on site a)
    y5: Theta_a-N (Adsorbed N on site a)
    Determining the steady state surface coverages allows to compute reaction rates.
"""

import matplotlib
matplotlib.use("TKAgg")
from matplotlib import pyplot as plt
import numpy as np 
import sksundae as sun

class KraehnertParameters:
    """Container for all parameters to pass to IDA"""
    def __init__(self, T, P, mole_fractions):
        self.T = T  # Temperature (K)
        self.P = P  # Pressure (kPa)
        self.pNH3 = mole_fractions['NH3'] * P
        self.pNO = mole_fractions['NO'] * P
        self.pO2 = mole_fractions['O2'] * P
        self.R = 8.314/1000  # kJ/(mol*K)
        self.surface_density = 2.72E-5  # mol/m^2
        self.T_ref = 385 + 273.15  # Reference temperature (K)

def compute_rates(data:KraehnertParameters, y:np.ndarray)->np.ndarray:
    """
    Compute reaction rates based on physical parameters and the steady state results of surface coverages
    returns an array of reaction rates: NH3, N2, NO, O2
    """
    R = data.R  # kJ/(mol*K)
    T = data.T  # Temperature in Kelvin
    T_ref = data.T_ref  # Reference temperature in Kelvin

    pNH3 = data.pNH3
    pNO = data.pNO
    pO2 = data.pO2
    
    E = np.zeros(10) # kJ/mol
    E[0] = 0
    E[1] = 60.9 
    E[2] = 0 
    E[3] = 181 
    E[4] = 99.5 
    E[5] = 154.8  
    E[6] = 63.5 
    E[7] = 139 
    E[8] = 135.4 
    E[9] = 155.2 

    # ki in mol m^-2 s^-1 kPa^-1 for reactions involving partial pressures
    k1 = 6.38E-1*np.exp(-(E[0]/R)*(1/T - 1/T_ref))
    k2 = 2.17E0*np.exp(-(E[1]/R)*(1/T - 1/T_ref))
    k3 = 2.94E-1*np.exp(-(E[2]/R)*(1/T - 1/T_ref))
    k4 = 1.09E-10*np.exp(-(E[3]/R)*(1/T - 1/T_ref))
    k5 = 5.91E2*np.exp(-(E[4]/R)*(1/T - 1/T_ref))
    k6 = 1.24E0*np.exp(-(E[5]/R)*(1/T - 1/T_ref))
    k7 = 2.63E-1*np.exp(-(E[6]/R)*(1/T - 1/T_ref))
    k8 = 6.42E1*np.exp(-(E[7]/R)*(1/T - 1/T_ref))
    k9 = 9.34E0*np.exp(-(E[8]/R)*(1/T - 1/T_ref))
    k10 = 5.2E0*np.exp(-(E[9]/R)*(1/T - 1/T_ref))

    R1 = k1*pNH3*y[0]
    R2 = k2*y[1]
    R3 = k3*pO2*y[2]**2
    R4 = k4*y[3]**2
    R5 = k5*y[1]*y[3]
    R6 = k6*y[4]
    R7 = k7*pNO*y[4]
    R8 = k8*y[5]**2
    R9 = k9*y[5]*y[3]
    R10 = k10*y[4]*y[5]

    R_NH3 = -R1 + R2 
    R_N2 = R8
    R_NO = R6 - R7 
    R_O2 = -R3 + R4 
    R_H2O = 1.5*R5
    R_N2O = R10
    return np.array([R_NH3, R_N2, R_NO, R_O2, R_H2O, R_N2O])


def residual_function(t, y, yp, res, userdata) -> None:
    """
    Kraehnert DAE system residual function.
    Equations are given in the same order as the original description in Krahenert's paper.
    The system is temperature and pressure dependent, with the temperature affecting the rate constants.
    Note that varying temperature and pressures affects teh jacobian as well, making the problem of variable stiffness
    Solving the problem as a DAE allows to easily enforce the site balances as algebraic equations.
    Reducing this index 1 system to an ODE system is possible and easy, but it results in a less stable system

    y0: Theta_b (free site b)
    y1: Theta_b-NH3 (Adsorbed NH3 on site b)
    y2: Theta_a (free site a)
    y3: Theta_a-O (Adsorbed O on site a)
    y4: Theta_a-NO (Adsorbed NO on site a)
    y5: Theta_a-N (Adsorbed N on site a)
    """
    user_data = userdata

    R = user_data.R  # kJ/(mol*K)
    surface_density = user_data.surface_density 

    T =  user_data.T # 385  # Temperature in Kelvin
    T_ref = user_data.T_ref  # Reference temperature in Kelvin
    
    E = np.zeros(10) # kJ/mol
    E[0] = 0
    E[1] = 60.9 
    E[2] = 0 
    E[3] = 181 
    E[4] = 99.5 
    E[5] = 154.8  
    E[6] = 63.5 
    E[7] = 139 
    E[8] = 135.4 
    E[9] = 155.2 

    # ki in mol m−2 s−1 kPa−1 for reactions involving partial pressures
    k = np.zeros(10)
    # Use absolute 0 as reference
    k[0] = 6.38E-1*np.exp(E[0]/(R*T_ref))
    k[1] = 2.17E0*np.exp(E[1]/(R*T_ref))
    k[2] = 2.94E-1*np.exp(E[2]/(R*T_ref))
    k[3] = 1.09E-10*np.exp(E[3]/(R*T_ref))
    k[4] = 5.91E2*np.exp(E[4]/(R*T_ref))
    k[5] = 1.24E0*np.exp(E[5]/(R*T_ref))
    k[6] = 2.63E-1*np.exp(E[6]/(R*T_ref))
    k[7] = 6.42E1*np.exp(E[7]/(R*T_ref))
    k[8] = 9.34E0*np.exp(E[8]/(R*T_ref))
    k[9] = 5.2E0*np.exp(E[9]/(R*T_ref))
    
    for i in range(len(k)):
        k[i] = k[i] * np.exp(-(E[i]/R)*(1/T))
                   
    pNH3 = user_data.pNH3
    pNO = user_data.pNO
    pO2 = user_data.pO2

    R = np.zeros(10)
    R[0] = k[0]*pNH3*y[0]
    R[1] = k[1]*y[1]
    R[2] = k[2]*pO2*y[2]**2
    R[3] = k[3]*y[3]**2
    R[4] = k[4]*y[1]*y[3]
    R[5] = k[5]*y[4]
    R[6] = k[6]*pNO*y[2]
    R[7] = k[7]*y[5]**2
    R[8] = k[8]*y[5]*y[3]
    R[9] = k[9]*y[4]*y[5]

    """
    Time derivatives of the DAE system variables. IDA expects the residuals in the var res, such that res = yp - f(t,y,yp,...)=0
    Surface coverages:
    y0: theta_b (free site b)
    y1: theta_b-NH3 (Adsorbed NH3 on site b)
    y2: theta_a (free site a)
    y3: theta_a-O (Adsorbed O on site a)
    y4: theta_a-NO (Adsorbed NO on site a)
    y5: theta_a-N (Adsorbed N on site a)
    yp_i are the time derivatives of the above variables
    """

    # Sundials wants the residuals in the form res_i = 0
    # keep in mind that res[0] and res[2] are algebraic equations (site balances)
    # site balance b
    res[0] = y[0] + y[1] - 1 
    # dtheta_b-NH3
    res[1] = yp[1] - (R[0] - R[1] - R[4])/surface_density
    # site balance a
    res[2] = y[2] + y[3] + y[4] + y[5] - 1 
    # dtheta-aO/dt
    res[3] = yp[3] - (2*R[2] - 2*R[3] - 1.5*R[4] - R[8])/surface_density
    # dthetha-aNO
    res[4] = yp[4] - (-R[5] + R[6] + R[8] - R[9])/surface_density
    # dtheta-aN
    res[5] = yp[5] - (R[4] - 2*R[7] - R[8] - R[9])/surface_density
    print(res)


def write_results(soln)->None:
    theta_b_NH3 = soln.y[-1,1]
    theta_a_O = soln.y[-1,3]
    theta_a_NO = soln.y[-1,4]
    theta_a_N = soln.y[-1,5]

    print(f"Theta_b-NH3: {theta_b_NH3:.6f}")
    print(f"Theta_a-O: {theta_a_O:.6f}")
    print(f"Theta_a-NO: {theta_a_NO:.6f}")
    print(f"Theta_a-N: {theta_a_N:.6f}")

    rates = compute_rates(params, soln.y[-1,:])

    print('selectivity N2O/NH3 %:', -2*rates[5]/rates[0]*100)
    
    print('rate of NH3', rates[0])
    print('rate of N2', rates[1])
    print('rate of NO', rates[2])
    print('rate of N2O', rates[5])




def plot_results(soln)->None:
    plt.figure()
    plt.semilogx(soln.t, soln.y[:,4])
    plt.xlabel('time')
    plt.ylabel("surface coverage");
    plt.show()
    plt.close('all')

    plt.figure()
    plt.semilogx(soln.t, soln.y[:,0] + soln.y[:,1], label='theta_b+theta_b-NH3')
    plt.show()
    plt.close()

    plt.figure()
    plt.semilogx(soln.t, soln.y[:,2] + soln.y[:,3] + soln.y[:,4] + soln.y[:,5], label='theta_a balance')
    plt.show()
    plt.close()


if __name__ == '__main__':
    # initial coverages. They must be consistent initial conditions for the coverages, i.e. satisfy the site balances.
    y0 = np.array([1 , 0.0, 1.0, 0.4, 0.0, 0.0])

    # Next is the derivatives. We don't pass consistent initial conditions for the derivatives. 
    # IDA computes them internally with the option calc_initcond='yp0'. 
    # In general, finding consistent initial conditions for DAE systems is not trivial.
    # The results have been validated against Fig 11 from the paper.

    yp0 = np.zeros_like(y0) 

    # time to simulate
    tspan = [0, 1E-1]

    params = KraehnertParameters(680+273.15, #K
                                 14, #kPa
                                 {'NH3':7/14, 'O2':7/14, 'NO':0.0})

    constraints = np.zeros_like(y0, dtype=int)

    solver = sun.ida.IDA(residual_function, atol=1e-8, algebraic_idx=[0,2], rtol=1e-4, userdata=params,
                        calc_initcond= 'yp0', max_order=2, constraints_idx=[1,3,4,5], 
                        constraints_type=[1,1,1,1])#, jacfn=jacobian_fn)
                        #, jacfn=jacobian_fn)

    soln = solver.solve(tspan, y0, yp0)

    write_results(soln)
    plot_results(soln)