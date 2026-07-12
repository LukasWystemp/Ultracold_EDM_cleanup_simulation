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
    




