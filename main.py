import numpy as np
from trajectory import Trajectory
import scipy as scp
from level_scheme import Level, LevelScheme, Laser
import level_scheme
import matplotlib.pyplot as plt
from photon_statistics import solve_moments, analytic_std

"""
Simulation: YbF molecule going through one laser pumping X^2\Sigma(v=0) -> A^2\Pi_{1/2}(v=0) with gaussian profile centred at y = mu and sigma std


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

def YbF_wavelength():
    """
    552 and 568 given by https://doi.org/10.48550/arXiv.1712.02868
    v2 and v3 have to be calculated from https://doi.org/10.48550/arXiv.1110.1868 with Dunham expansion parameteres https://en.wikipedia.org/wiki/Dunham_expansion
    Assume D_e', B_e', D_e', D_e'' << \omega_e'' = 506.674, \omega_e x'' = 2.245 and offset cancels
    E = T_e + \omega_e (v + 1/2) - \omega_e x (v + 1/2)^2
    \v_{0i} = T_e' + 1/2 \omega' - 1/4 \omega' x' - \omega''(v + 1/2) + \omega x'' (v + 1/2)^2
    use \v_{00} = 18106 cm^-1 to find T_e' = 18091 cm^-1
    \v_{01} = 17604 cm^-1, \v_{02} = 17091.68, \v_{03} = 16593.25
    \lambda_{0i} = 10^7 / \v_{0i} 
    """
    return {Level.g: 552e-9, Level.v1: 568e-9, Level.v2: 585e-9, Level.v3: 602e-9}


def main():
    tau = 20e-5
    vy = 170 # ms^-1

    mu = 0.5 * vy * tau # centre
    sigma = 0.00245 # m
    I0 = 210 # Wm^-2

    dr = 1/(28e-9)  # https://doi.org/10.1039/c1cp21585j # decay rate for e state
    YbF = LevelScheme(decay_rate = dr, 
                        vibrational_branching = {Level.g: 0.9307, Level.v1: 0.066, Level.v2: 0.003, Level.v3: 0.0003}, 
                        wavelength=YbF_wavelength(), 
                        degeneracy_factor = {Level.g: 12, Level.v1: 12, Level.v2: 12, Level.v3: 12, Level.e: 4})

    YbF.print_isat()
    laser = Laser(target_v = Level.g, detuning = 0, wavelength=552e-9, mu =mu, sigma=sigma, I0 = I0) # P(1), I0 in Wm^-2

    p1 = Trajectory(v_y = vy, x_0 = 0, z_0 = 0, N0 = [1.0,0.0,0.0,0.0,0.0,0.0])
    sol = solve_transit(YbF, laser, p1, tau)
    print(sol.message, " n_eval:", sol.t.size)
    
    M_mid = p1.build_rate_matrix(tau / 2, YbF, laser)
    col_sums = M_mid[:5, :5].sum(axis=0) # exclude photon_count row, one-way accumulator
    print("Column sums of M at t=tau/2 (should be ~0):", col_sums)


    n_photons_final = sol.y[5, -1]
    print(f"Photons radiated over transit: {n_photons_final:.3f}")

    #plot_results(sol, YbF, laser, p1, tau, dr, n_photons_final)

    sol_moments = solve_moments(YbF, laser, p1, tau)
    t = np.linspace(0, tau, 8000)
    mean_analytic, std_analytic = analytic_std(sol_moments, t)
    print(mean_analytic, std_analytic)
    plot_photon_statistics(t, mean_analytic, std_analytic)


def plot_photon_statistics(t, mean_analytic, std_analytic):
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
 
    #ax = axes[0]
    ax.plot(t * 1e6, mean_analytic, color='tab:blue', label='mean photon count')
    ax.fill_between(t * 1e6, mean_analytic - std_analytic, mean_analytic + std_analytic, color='tab:blue', alpha=0.25,
                     label=r'$\pm 1\sigma$ (exact, analytic)')
    ax.axhline(14.9, color='k', linestyle=':', label='lit. average (14.9)')
    ax.set_ylabel("photon count")
    ax.set_xlabel("time (microseconds)")
    ax.set_title("Exact mean $\\pm$ 1 std.dev of photon count (moment-hierarchy ODE)")
    ax.legend(loc='upper left')
 
 
def plot_results(sol, level_scheme, laser, trajectory, tau, dr, n_photons_final):
    t_plot = np.linspace(0, tau, 2000)
    N = sol.sol(t_plot)
 
    labels = ["v=0", "v=1", "v=2", "v=3", "excited"]
 
    y = trajectory.v_y * t_plot
    I_profile = laser.profile(trajectory.x_0, y, trajectory.z_0)
 
    fig, axes = plt.subplots(5, 1, figsize=(8, 12), sharex=True)
 
    ax = axes[0]
    colours = ["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:red"]
    for i in range(5):
        ax.plot(t_plot * 1e6, N[i], label=labels[i], color=colours[i])
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
    ax2.plot(t_plot * 1e6, I_profile, color='gray', linestyle='--', alpha=0.6, label="laser intensity")
    ax2.set_ylabel(r"laser intensity (unnormalized) / $\text{Wm}^{-2}$", color='gray')
    ax.set_title("Excited-state population vs. laser intensity envelope")

    ax = axes[2]
    ax.plot(t_plot * 1e6, N[5], color='tab:purple', label="photons radiated")
    ax.set_ylabel("cumulative photon count")
    ax.set_title("Radiation profile: Total photons emitted")
    ax.text(0.85, 0.15, f"Total photons: {n_photons_final:.2f}", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=dict(facecolor='tab:purple', alpha=0.5))

    ax = axes[3]
    ax.plot(t_plot*1e6, dr*N[4]*1e-6, color='tab:purple', label="Photon emission")
    ax.set_ylabel("Photon emission")
    ax.set_title("Radiation profile: photons emitted per unit time")
 
    ax = axes[4]
    total = N[:5].sum(axis=0)
    ax.plot(t_plot * 1e6, total - 1.0)
    ax.set_ylabel("total population - 1\n(conservation check)")
    ax.set_xlabel("time (microseconds)")
    ax.set_title("Population conservation check")
 
    fig.tight_layout()
    plt.show()
 


if __name__=="__main__":
    print("Simulation ")
    main()