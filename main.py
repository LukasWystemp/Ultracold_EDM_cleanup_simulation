import numpy as np
from trajectory import Trajectory
import scipy as scp
from level_scheme import Level, LevelScheme, Laser
import level_scheme
import matplotlib.pyplot as plt
from photon_statistics import solve_moments, analytic_std, mc_ensemble, pmt

"""
Simulation: YbF molecule going through one laser pumping X^2\Sigma(v=0) -> A^2\Pi_{1/2}(v=0) with gaussian profile centred at y = mu and sigma std

To-Do
Find PMT efficiency and solid angle

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


def plot_photon_statistics(t, mean_analytic, std_analytic, n_molecules, counts, mc_avg, mc_std, all_photon_times, pmt_bins, pmt_counts):
    fig, axes = plt.subplots(3, 1, figsize=(8, 12))
 
    ax = axes[0]
    ax.plot(t * 1e6, mean_analytic, color='tab:blue', label='mean photon count')
    ax.fill_between(t * 1e6, mean_analytic - std_analytic, mean_analytic + std_analytic, color='tab:blue', alpha=0.25,
                     label=r'$\pm 1\sigma$ (exact, analytic)')
    ax.axhline(14.9, color='k', linestyle=':', label='lit. average (14.9)')
    ax.set_ylabel("photon count")
    ax.set_xlabel("time (microseconds)")
    ax.set_title("Exact mean $\\pm$ 1 std.dev of photon count (moment-hierarchy ODE)")
    ax.legend(loc='upper left')

    ax = axes[1]
    max_n = int(counts.max()) + 2
    ax.hist(counts, bins=np.arange(0, max_n) - 0.5, color='tab:green', alpha=0.7,
            label=f'Monte Carlo ({n_molecules} molecules)')
    ax.axvline(mc_avg, color='red', linestyle='--',linewidth=2, label=f'MC Mean ({mc_avg:.2f})')
    ax.axvline(14.9, color='k', linestyle=':', label='lit. average (14.9)')
    ax.axvline(mc_avg - mc_std, color='tab:blue', linestyle='--')
    ax.axvline(mc_avg + mc_std, color='tab:blue', linestyle='--', label=f'MC $\\pm 1\\sigma ({mc_std:.2f})$')
    ax.set_xlabel("total photons emitted per molecule")
    ax.set_ylabel("number of molecules")
    ax.set_title("Full photon-count distribution")
    ax.legend()

    ax = axes[2]
    pmt_t = 0.5 * (pmt_bins[:-1] + pmt_bins[1:])
    ax.errorbar(pmt_t * 1e6, pmt_counts, yerr=np.sqrt(pmt_counts), fmt='.', color='tab:purple',
                 alpha=0.6, label=f'ensemble PMT counts/bin')
    ax.set_ylabel("detected counts / bin\n(ensemble)")
    ax.set_xlabel("time (microseconds)")
    ax.set_title("Synthetic PMT: Ensemble rate with shot noise")
    ax.legend(loc='upper right')

    fig.tight_layout()
 
 
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

    plot_results(sol, YbF, laser, p1, tau, dr, n_photons_final)

    # Moments
    sol_moments = solve_moments(YbF, laser, p1, tau)
    t = np.linspace(0, tau, 8000)
    mean_analytic, std_analytic = analytic_std(sol_moments, t)

    # MC
    p2 = Trajectory(v_y = vy, x_0=0, z_0=0, N0=None)
    
    n_molecules = 8000
    all_photon_times, counts = mc_ensemble(YbF, laser, p2, tau, n_molecules)
    print(f"MC ({n_molecules} molecules): mean={counts.mean():.3f}, std={counts.std():.3f}")

    epsilon = 0.2 # Efficiency * solid_angle / 2\pi # FIND ACTUAL VALUE
    bins, pmt_counts = pmt(all_photon_times, tau, epsilon)

    plot_photon_statistics(t, mean_analytic, std_analytic, n_molecules, counts, counts.mean(), counts.std(), all_photon_times, bins, pmt_counts)




if __name__=="__main__":
    print("Simulation ")
    main()