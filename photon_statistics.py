import numpy as np
import scipy as scp
from level_scheme import Level

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


def get_max_pump_rate(level_scheme, laser, trajectory, tau):
    ts = np.linspace(0, tau, 10000)
    pump_rates = [abs(trajectory.get_pump_rate(t, laser, level_scheme)) for t in ts]
    return max(pump_rates)


def simulate_mc(level_scheme, laser, trajectory, tau, rng):
    Gamma = level_scheme.decay_rate
    t = 0.
    dt = (tau - t) / 2000
    state = Level.g

    #max_pump_rate = get_max_pump_rate(level_scheme, laser, trajectory, tau)
    
    photon_times = []
    while t < tau:
        if state in (Level.v1, Level.v2, Level.v3):
            break

        if state == Level.g:
            pr = abs(trajectory.get_pump_rate(t, laser, level_scheme))
            if rng.uniform(0,1) < pr*dt:
                state = Level.e
        
        elif state == Level.e:
            if rng.uniform(0,1) < Gamma*dt:
                photon_times.append(t)
                _, probs = zip(*level_scheme.vibrational_branching.items())
                state = rng.choice(np.array([Level.g,Level.v1,Level.v2,Level.v3]) , p=probs)
        t += dt
    
    return photon_times
    

def mc_ensemble(level_scheme, laser, trajectory, tau, n_molecules):
    rng = np.random.default_rng(12345)

    all_photon_times = []
    counts = np.zeros(n_molecules)
    for i in range(n_molecules):
        pts = simulate_mc(level_scheme, laser, trajectory, tau, rng)
        all_photon_times.append(pts)
        counts[i] = len(pts)
    return all_photon_times, counts



def pmt(all_photon_times, tau, epsilon=1.0):
    bin_width = 5e-6
    bins = np.arange(0, tau + bin_width, bin_width)

    counts = np.zeros(len(bins) - 1)
    for pts in all_photon_times:
        if (epsilon < 1.):
            pts = np.array(pts)
            pts = pts[np.random.uniform(size=len(pts)) < epsilon]
        counts += np.histogram(pts, bins=bins)[0]
    return bins, counts




