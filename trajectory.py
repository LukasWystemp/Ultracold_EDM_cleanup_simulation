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

        Lambda = np.zeros(4) # pumping rate

        I = laser.profile(self.x_0, y, self.z_0)
        s = I / level_scheme.isat[laser.target_v]
        s *= level_scheme.degeneracy_factor[laser.target_v]
        delta = laser.detuning
        gamma = level_scheme.decay_rate # / 2 Not sure if /2 is needed. Double check
        Lambda[laser.target_v.value] = (level_scheme.decay_rate / 2)*s / (1 + (2*delta/level_scheme.decay_rate)**2)
        

        M = np.zeros((5,5)) # order: v0,v1,v2,v3,excited
        for i, (level, vb) in enumerate(q.items()):
            M[i, i]   -= Lambda[i]
            M[i, 4]   += gamma * vb
            M[4, i]   += Lambda[i]
        M[4, 4] -= gamma
        return M
    




