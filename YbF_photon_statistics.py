"""
Minimal skeleton: full YbF hyperfine (F, m_F) structure in pyLCP.

"""
import numpy as np
import scipy.constants as cts
import pylcp
from pylcp.hamiltonians import XFmolecules
import matplotlib.pyplot as plt

# ---------------------------------------------
# Unit system
tau = 28e-9 # lifetime
gamma_MHz = 1/(2*np.pi*tau)/1e6 # linewidth (energy unit)
Gamma = 1/tau # angular linewidth (rad/s)

w_p1 = 552e-9
w_v1 = 568e-9
k_p1 = 2*np.pi/w_p1 # p1 wavevector
k_v1 = w_p1 / w_v1 # v1 wavevector in units of k_p1

x0 = 1/k_p1 # unit of distance
t0 = tau # unit of time
v_unit = x0 / t0

mm = 1e-3/x0 # 1mm in pylcp units

fcf = np.array([0.9307, 0.066, 0.003, 0.0003])
mass = 193*cts.value('atomic mass constant')/(cts.hbar*k_p1**2*tau)
 
z_laser = 5*mm
w_sigma_m = 2.5e-3 # in meter
w_beam = w_sigma_m*1e3*mm

det_p1, det_v1 = 0.0, 0.0 # detuning
P_p1 = 120e-3
P_v1 = 38e-3

# --------------------------------------------------------------------
# X^2Sigma+ (N=1, v=0,1,2,3)
H0_X, Bq_X, U_X, X_bases = {}, {}, {}, {}
for v in range(4):
    H0, Bq, U, Xbasis = XFmolecules.Xstate(
        N=1,
        I=1/2,
        B=7233.69, # rotational constant https://arxiv.org/pdf/2306.05563
        gamma= -13.424, # spin-rotation constant https://doi.org/10.1103/PhysRevLett.74.1554
        b=141.7956, # Frosch-Foley isotropic hyperfine const. https://arxiv.org/pdf/2306.05563
        c=85.4026, # Frosch-Foley anisotropic hyperfine const. https://arxiv.org/pdf/2306.05563
        CI=0.02038, #nuclear-spin-rotation
        gS=2.0023193043622,         # free-electron g-factor
        gI=5.26,                    # 19F nuclear g-factor
        muB=cts.value('Bohr magneton in Hz/T') / 1e6 * 1e-4,
        return_basis=True,
    )
    H0_X[v], Bq_X[v], U_X[v], X_bases[v] = H0, Bq, U, Xbasis
    print(f"X(v={v}, N=1): {H0.shape[0]} (F, m_F) states, "
          f"hyperfine energies (MHz): {np.round(np.unique(np.diag(H0)), 2)}")
 
n_X = H0_X[0].shape[0] 

# ---------------------------------------------
# Excited state: A^2Pi_1/2 (v'=0, J'=1/2)
H0_A, Bq_A, Abasis = XFmolecules.Astate(
    J=1/2,
    I=1/2,
    P=+1, # parity
    a=55.3, b=531.48,c=63.69, # hpyerfine constants
    glprime=-0.05, # A-state Lambda-doubling/orbital g-factor # Check!
    gS=2.0023193043622,
    muB=cts.value('Bohr magneton in Hz/T') / 1e6 * 1e-4,
    return_basis=True,
)
n_A = H0_A.shape[0]  
print(f"A(v'=0, J'=1/2): {n_A} (F', m_F') states, "
      f"hyperfine energies (MHz): {np.round(np.unique(np.diag(H0_A)), 2)}")
 

# --------------------------------------------------------
# Build Hamiltonian
dijq_raw = XFmolecules.dipoleXandAstates(X_bases[0], Abasis, I=1/2, S=1/2)
dijq = {v: np.einsum('ij,qjk->qik', U_X[v].T, dijq_raw) for v in range(4)}

ham = pylcp.hamiltonian(mass=mass)
for v in range(4):
    ham.add_H_0_block(f'X{v}', H0_X[v]/gamma_MHz)
    ham.add_mu_q_block(f'X{v}', Bq_X[v]/gamma_MHz)
ham.add_H_0_block('A', H0_A/gamma_MHz)
ham.add_mu_q_block('A', Bq_A/gamma_MHz)

ham.add_d_q_block('X0', 'A', np.sqrt(fcf[0])*dijq[0], k=1.)
ham.add_d_q_block('X1', 'A', np.sqrt(fcf[1])*dijq[1], k=k_v1)
ham.add_d_q_block('X2', 'A', np.sqrt(fcf[2])*dijq[2], k=w_p1/583e-9)
ham.add_d_q_block('X3', 'A', np.sqrt(fcf[3])*dijq[3], k=w_p1/600e-9)
 
ham.print_structure()
 
# ------------------------------------------------
# Laser beams

def power_to_s(P_total, n_sidebands, sigma_m, lambd):
    I_sat = np.pi*cts.h*cts.c/(3*lambd**3*tau)
    I0 = (P_total/n_sidebands)/(2*np.pi*sigma_m**2)
    return I0/I_sat

def crossed_beam_s(s_max, wb, zc):
    # return callable with signature (R,t)
    def s(R, t):
        return s_max*np.exp(-0.5*(R[1]**2 + (R[2]-zc)**2)/wb**2)
    return s

def beam_pair(k, s_max, deltas):
    beams = []
    for delta in deltas:
        for sign in (+1, -1):
            beams.append({'kvec':np.array([sign*k, 0., 0.]),
                          'pol': +1, 
                          'delta': delta,
                          's': crossed_beam_s(s_max, w_beam, z_laser)})
    return pylcp.laserBeams(beams)

# Beams on resonance. pyLCP measures delta relative to zero-point of H block. 
# Resonance for |X, F> -> |A, F'> occurs at delta = E_F' - E_F
E_X_hf = np.unique(np.diag(H0_X[0]))/gamma_MHz # 4 F levels
E_A_F1 = np.max(np.diag(H0_A))/gamma_MHz  # F' = 1

deltas_p1 = E_A_F1 - E_X_hf + det_p1
deltas_v1 = E_A_F1 - E_X_hf + det_v1

s_p1 = power_to_s(P_p1, 4, w_sigma_m, w_p1)
s_v1 = power_to_s(P_v1, 4, w_sigma_m, w_v1)
print(f"peak s per sideband: main {s_p1:.4f}, repump {s_v1:.4f}")

laserBeams = {
    'X0->A': beam_pair(1., s_p1, deltas_p1),
    'X1->A': beam_pair(k_v1, s_v1, deltas_v1),
}

B_remix = 0.7 # Gauss
magField = pylcp.constantMagneticField(B_remix*np.array([0., 1., 1.])/np.sqrt(2))
 
# ----------------------------------
# Rate equation
rateeq = pylcp.rateeq(laserBeams, magField, ham, include_mag_forces=False)
 
v_forward = 170./v_unit
rateeq.set_initial_position_and_velocity(np.array([0., 0., 0.]), np.array([0., 0., v_forward]))

n_states = 4*n_X + n_A
pop0 = np.zeros(n_states)
pop0[:n_X] = 1./n_X
rateeq.set_initial_pop(pop0)

t_max = (15*mm)/v_forward
sol = rateeq.evolve_motion([0., t_max], t_eval=np.linspace(0., t_max, 500),
                           rtol=1e-8, atol=1e-10)
 
# ---------------------------------------------------
# Photon statistics
pops = sol.N 
P_X = [pops[v*n_X:(v+1)*n_X].sum(axis=0) for v in range(4)]
P_A = pops[4*n_X:4*n_X+n_A].sum(axis=0)

def gaussian(z):
    return np.exp(-0.5*(((z-z_laser)/v_forward)**2)/w_beam**2)

t = np.linspace(0, t_max, 500)
for v in range(4):
    plt.plot(t, P_X[v], label=f"v{v}")
plt.plot(t, P_A, label="excited")
plt.plot(t, gaussian(t), label="laser")
plt.legend()

plt.show()

n_photons = np.trapz(P_A, sol.t)
print("\n--- deterministic rate equation ---")
for v in range(4):
    print(f"final population in X(v={v}): {P_X[v][-1]:.4f}")
print(f"final population in A: {P_A[-1]:.2e}")
print(f"total scattered photons: {n_photons:.1f}")

# ---------------------------------------------------
# fukc

n_seg = 400
t_edges = np.linspace(0., t_max, n_seg + 1)


