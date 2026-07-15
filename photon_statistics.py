import numpy as np
import scipy as scp
from level_scheme import Level
import time

def rhs_moments(t, psi, particle, level_scheme, lasers_list):
    V = particle.build_moment_matrix(t, level_scheme, lasers_list)
    return V @ psi


def solve_moments(level_scheme, lasers_list, particle, tau):
    psi0 = np.zeros(15) # P, P1, P2
    psi0[:5] = particle.N0[:5] # Population P, P1 = P2 = 0 at t=0
    sol = scp.integrate.solve_ivp(
        rhs_moments, [0, tau], psi0,
        args=(particle, level_scheme, lasers_list),
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

"""
def get_max_pump_rates(level_scheme, lasers_list, trajectory, tau):
    ts = np.linspace(0, tau, 1000000)
    pump_rates = [np.abs(trajectory.get_pump_rates(t, lasers_list, level_scheme)) for t in ts]
    pump_rates = np.array(pump_rates)
    return np.max(pump_rates, axis=0)
"""
def get_max_pump_rates(level_scheme, lasers_list, trajectory, tau):
    max_rates = np.zeros(4)
    for laser in lasers_list:
        t_peak = np.clip(laser.mu / trajectory.v_y, 0, tau)
        rates_at_peak = np.abs(trajectory.get_pump_rates(t_peak, lasers_list, level_scheme))
        i = laser.target_v.value
        max_rates[i] = rates_at_peak[i]
    return max_rates


def simulate_mc(level_scheme, lasers_list, trajectory, tau, rng, max_rate):
    Gamma = level_scheme.decay_rate
    t = 0.
    #dt = (tau - t) / 20000
    state = Level.g

    targeted_states = {laser.target_v for laser in lasers_list}
    
    photon_times = []
    while t < tau:
        if state == Level.e:
            t += rng.exponential(1 / Gamma)
            if t >= tau:
                break
            photon_times.append(t)
            _, probs = zip(*level_scheme.vibrational_branching.items())
            state = rng.choice(np.array([Level.g,Level.v1,Level.v2,Level.v3]) , p=probs)
        else: # g, v1, v2, v3
            if state not in targeted_states:
                break

            accepted = False
            while not accepted and t < tau:
                mr = max_rate[state.value]
                t += rng.exponential(1 / mr)
                if t >= tau:
                    break
                pump_rates = abs(trajectory.get_pump_rates(t, lasers_list, level_scheme))
                pr = abs(pump_rates[state.value])
                if rng.uniform(0,1) < pr / mr:
                    accepted = True
            if accepted:
                state = Level.e

    return photon_times
    

def mc_ensemble(level_scheme, lasers_list, trajectory, tau, n_molecules):
    rng = np.random.default_rng(12345)

    all_photon_times = []
    counts = np.zeros(n_molecules)
    max_rate = get_max_pump_rates(level_scheme, lasers_list, trajectory, tau)
    for i in range(n_molecules):
        pts = simulate_mc(level_scheme, lasers_list, trajectory, tau, rng, max_rate)
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




