import numpy as np
from dataclasses import dataclass
from level_scheme import Level

@dataclass
class Trajectory:
    v_y: float
    x_0: float
    z_0: float
    N0: np.ndarray

    def build_rate_matrix(self, t, level_scheme, laser):
        y = self.v_y * t
        q = level_scheme.vibrational_branching

        Lambda = np.zeros(5) # pumping rate

        Gamma = level_scheme.decay_rate

        I = laser.profile(self.x_0, y, self.z_0)
        
        I_eff = (2 * level_scheme.degeneracy_factor[laser.target_v]**2 /(level_scheme.degeneracy_factor[laser.target_v] + level_scheme.degeneracy_factor[Level.e])) * level_scheme.isat[laser.target_v]
        s = I / I_eff

        delta = laser.detuning
        
        # https://doi.org/10.48550/arXiv.2510.16203
        pre_fac = level_scheme.degeneracy_factor[Level.e] / (level_scheme.degeneracy_factor[laser.target_v] + level_scheme.degeneracy_factor[Level.e] )
        Lambda[laser.target_v.value] = Gamma * pre_fac * s / (1 + s + (2*delta/Gamma)**2)
        

        M = np.zeros((6,6)) # order: v0,v1,v2,v3,excited, photon_counter
        for i, (level, vb) in enumerate(q.items()):
            M[i, i]   -= Lambda[i]
            M[i, 4]   += Gamma * vb
            M[4, i]   += Lambda[i]
        M[4, 4] -= Gamma

        M[5, 4] += Gamma # photon emission is Gamma * N_e, only inflow, no outflow
        return M 
    
    def get_pump_rate(self, t, laser, level_scheme):
        i = laser.target_v.value
        M = self.build_rate_matrix(t, level_scheme, laser)
        return M[i, i]
    
    def build_moment_matrix(self, t, level_scheme, laser):
        """
        Let p_i(n, t) = Pr(state(t) = i, N(t) = n)
        dp_i(n,t)/dt = - \sum_k R_{i->k} p_i(n,t) + \sum_{j \in nc} R_{j->i} p_j(n,t) + \sum_{j \in c} R_{j->i} p_j(n-1,t)
        P_i(t) = \sum_n p_i(n,t) is the population vector
        P1_i(t) = E(N 1_i) = \sum_n n p_i(n,t) and P2_i(t) = E(N(N-1) 1_i) = \sum_n n(n-1) p_i(n,t) are the first and second moments
        For P1, multiply by n and sum over n, then re-index last term, similar for P2 with n(n-1)
        dP/dt = M P 
        dP1(t)/dt = M P1 + \sum_{j \in c} R_{j->i} P_j(t)
        dP2(t)/dt = M P2 + 2 \sum_j{j->i} P1_j(t)
        

        Define C = R_{j->i}
        """
        M = self.build_rate_matrix(t, level_scheme, laser)
        M = M[:5, :5]
        Gamma = level_scheme.decay_rate
        C = np.zeros((5,5))
        for i, (level, vb) in enumerate(level_scheme.vibrational_branching.items()):
            C[i, 4] = Gamma * vb
        
        Z = np.zeros((5,5))
        V = np.block([[M, Z, Z], 
                     [C, M, Z],
                     [Z, 2*C, M]])
        return V




