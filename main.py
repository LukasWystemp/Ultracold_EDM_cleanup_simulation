import numpy as np
from trajectory import Trajectory
import scipy as scp
from level_scheme import Level, LevelScheme, Laser
import matplotlib.pyplot as plt

"""
Simulation: YbF molecule going through one laser pumping X^2\Sigma(v=0) -> A^2\Pi_{1/2}(v=0) with gaussian profile centred at y = mu and sigma std


To Do: 
Confirm values
- detuning
- wavelengths
- mu, sigma i.e. laser profile
- degeneracy factors (Somehow dependant on hyperfine + Glebsch-Gordon)

Confirm physics
- How do degeneracy factors affect saturation intensity
- Is Frank-Condon factor up (v1 -> v0) affected by vibrational branching ratio?
- Relationship decay rate of spontaneous emission and gamma  (gamma = dr / 2)? [Removed /2 for now]
- Lambda = (gamma*s) / (1 + s + (2*delta*gamma)**2 ) OR (gamma*s) / (1 + (2*delta*gamma)**2 )?

Simulation
- Time resolution? 
- Add doppler shift

"""

def rhs(t, N, trajectory, level_scheme, laser):
    M = trajectory.build_rate_matrix(t, level_scheme, laser)
    return M @ N


def solve_transit(level_scheme, laser, particle, tau):
    sol = scp.integrate.solve_ivp(
        rhs, [0, tau], particle.N0,
        args=(particle, level_scheme, laser),
        method='Radau',
        rtol=1e-8, atol=1e-14,
        dense_output=True
    )
    return sol

def main():
    tau = 6e-5
    vy = 170

    mu = 0.5 * vy * tau
    sigma = 0.08 * vy * tau

    dr = 1/(28e-9)  # https://doi.org/10.1039/c1cp21585j # decay rate for e state
    YbF = LevelScheme(decay_rate = dr, 
                        vibrational_branching = {Level.g: 0.932, Level.v1: 0.065, Level.v2: 0.00299, Level.v3: 0.00001}, 
                        wavelength={Level.g: 552e-9, Level.v1: 568e-9, Level.v2: 584e-9, Level.v3: 600e-9}, # NEED TO FIND ACTUAL VALUES
                        degeneracy_factor = {Level.g: 1, Level.v1: 1, Level.v2: 1, Level.v3: 1, Level.e: 1})

    laser = Laser(target_v = Level.g, detuning = 6.5e6, wavelength=552e-9, mu =mu, sigma=sigma) # FIND REAL LASER PARAMETERS mu sigma

    p1 = Trajectory(v_y = vy, x_0 = 0, z_0 = 0, N0 = [1.0,0.0,0.0,0.0,0.0])
    sol = solve_transit(YbF, laser, p1, tau)
    print(sol.message, " n_eval:", sol.t.size)
    
    M_mid = p1.build_rate_matrix(tau / 2, YbF, laser)
    col_sums = M_mid.sum(axis=0)
    print("Column sums of M at t=tau/2 (should be ~0):", col_sums)
 
    plot_results(sol, YbF, laser, p1, tau)
 
 
def plot_results(sol, level_scheme, laser, trajectory, tau):
    t_plot = np.linspace(0, tau, 2000)
    N = sol.sol(t_plot)
 
    labels = ["v=0", "v=1", "v=2", "v=3", "excited"]
 
    y = trajectory.v_y * t_plot
    I_profile = laser.profile(trajectory.x_0, y, trajectory.z_0)
    print(I_profile)
 
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
 
    ax = axes[0]
    for i in range(5):
        ax.plot(t_plot * 1e6, N[i], label=labels[i])
    ax.set_yscale('log')
    ax.set_ylabel("ground-state population")
    ax.set_ylim(1e-6, 2)
    ax.legend(loc='center right')
    ax.set_title("Ground vibrational-level populations")
 
    ax = axes[1]
    ax.plot(t_plot * 1e6, N[4], color='tab:red', label="N_e")
    ax.set_ylabel("excited-state population", color='tab:red')
    ax.tick_params(axis='y', labelcolor='tab:red')
    ax2 = ax.twinx()
    ax2.plot(t_plot * 1e6, I_profile, color='gray', linestyle='--', alpha=0.6, label="laser intensity (norm.)")
    ax2.set_ylabel("laser intensity (unnormalized)", color='gray')
    ax.set_title("Excited-state population vs. laser intensity envelope")
 
    ax = axes[2]
    total = N.sum(axis=0)
    ax.plot(t_plot * 1e6, total - 1.0)
    ax.set_ylabel("total population - 1\n(conservation check)")
    ax.set_xlabel("time (microseconds)")
    ax.set_title("Population conservation check")
 
    fig.tight_layout()
 


if __name__=="__main__":
    print("Simulation ")
    main()