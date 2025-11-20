import numpy as np
from scipy.stats import qmc
from Krahnert import Krahnert
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm

np.random.seed(1)

"""
Parallelized latin hypercube sampling of the Krahnert mechanism over a defined parametric space.
Why hypercube sampling? Because random sampling in many dimensions tends to cluster, so it doesn't scan the parametric 
space effectively unless you increase the amount of data, which is not efficient.
This work has no scaling in the parameters.
Typical scaling are T -> 1/T and x_i -> ln(x_i) which are physically motivated
Especially relevant is the scaling of the outputs scaled as ln(r_i) performs best, which will be shown later

Explores the 4 dimensional parametric space at fix pressure of 500kPa. 
Code is pallelized to the maximum number of available processors
Typical performance is about 14 ms/it on 6 processors
"""

def solve_single(params_row):
    """
    Takes dict of parameters, returns combined dict with rates
    Handles both dict and numpy array outputs
    """
    kra = Krahnert(T=params_row[0], P=500e3, mole_fractions={'NH3':params_row[1],
                                                            'O2': params_row[2],
                                                            'NO': params_row[3]})
                
    # Calculate rates (handles both array and dict returns)
    try:
        rates = kra.solve()
    except RuntimeError:
        # Increase relaxation in case of failure
        rates = kra.solve(relaxation=1e4)
    return rates
        

if __name__ == '__main__':

    param_bounds = {
        'T': (500, 1500),      # K
        'xNH3': (0.001, 0.2),     
        'xO2': (0.001, 0.2),      
        'xNO': (0.001, 0.2),      
    }

    n_cpu = 6
    n_samples = 100_000
    dim = len(param_bounds)

    sampler = qmc.LatinHypercube(
        d=dim, 
        scramble=True,  
        rng = 1,      # fix the seed for reproducibility
        optimization=None     
    )

    samples = sampler.random(n=n_samples)

    scaled_samples = np.zeros_like(samples)
    for i, (param_name, (low, high)) in enumerate(param_bounds.items()):
        scaled_samples[:, i] = low + samples[:, i] * (high - low)

    with Pool(processes=n_cpu) as pool:
        results = list(tqdm(
            pool.imap(solve_single, scaled_samples),
            total=len(scaled_samples),
            desc="Solving with rates"
        ))
    
    df_rates = pd.DataFrame(results)
    df_rates['T'] = scaled_samples[:,0]
    df_rates['xNH3'] = scaled_samples[:,1]
    df_rates['xO2'] = scaled_samples[:,2]
    df_rates['xNO'] = scaled_samples[:,3]

    df_rates.to_csv('rates.csv', index=False)

