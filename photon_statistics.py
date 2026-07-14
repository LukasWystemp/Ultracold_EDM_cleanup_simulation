import numpy as np
import scipy as scp

def rhs_moments(t, psi, particle, level_scheme, laser):
    V = particle.build_moment_matrix(t, level_scheme, laser)
    return V @ psi


def solve_moments(level_scheme, laser, particle, tau):
    psi0 = np.zeros(15) # P, P1, P2
    psi0[:5] = particle.N0[:5] # Population P, P1 = P2 = 0 at t=0
    sol = scp.integrate.solve_ivp(
        rhs_moments, [0, tau], psi0,
        args=(particle, level_scheme, laser),
        method='Radau', rtol=1e-9, atol=1e-14,
        dense_output=True,
    )
    return sol

def analytic_std(sol, t):
    Psi = sol.sol(t)
    P1 = Psi[5:10]
    P2 = Psi[10:15]
    mean = P1.sum(axis=0) # E(X)
    E_xxm1 = P2.sum(axis=0) # E(X(X-1))
    var = E_xxm1 + mean - mean**2
    return mean, np.sqrt(var)

