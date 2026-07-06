import numpy as np
from dataclasses import dataclass, field
from enum import Enum
import scipy as scp

@dataclass
class LevelScheme:
    vibrational_branching: dict # Frank Condon
    isat: dict # {v: I_sat}
    wavelength: dict
    degeneracy_factor: dict # {v: n}
    decay_rate: float # e state

    def __init__(self, decay_rate, vibrational_branching, wavelength, degeneracy_factor):
        self.decay_rate = decay_rate
        self.vibrational_branching = vibrational_branching
        self.wavelength = wavelength
        self.degeneracy_factor = degeneracy_factor

        self.__post_init__()

    def __post_init__(self):
        self.isat = self.compute_isat()


    def compute_isat(self): # 10.1088/1367-2630/15/5/053034
        # Note to self: ask if this is correct, there are contradicting formulas on the internet abt this
        isat = {}

        for level, vb in self.vibrational_branching.items():
            # need to adjust for FC factor with self.decay_rate * vb? 
            w = self.wavelength[level]
            n = self.degeneracy_factor[level]
            isat[level] = n * np.pi * scp.constants.c * scp.constants.h * self.decay_rate / (3 *w**3 * vb) # need n? 

        return isat
    
    def print_isat(self):
        print(self.isat)


class Level(Enum):
    g = 0
    v1 = 1
    v2 = 2
    v3 = 3
    e = 4


@dataclass
class Laser:
    target_v: Level
    detuning: float
    wavelength: float


    mu: float
    sigma: float
    profile: callable = field(init=False, repr=False)

    def __post_init__(self):
        def tophat(x, y, z):
            # propagating along x
            I0 = 1 / (self.sigma * np.sqrt(2 * np.pi)) / 10

            y = np.asarray(y)
            edge_width = 0.01 * self.sigma 

            left_edge = 1 / (1 + np.exp(-(y - (self.mu - self.sigma)) / edge_width))
            right_edge = 1 / (1 + np.exp((y - (self.mu + self.sigma)) / edge_width))

            return I0 * left_edge * right_edge
        
        def gaussian(x, y, z):
            # propagating along x
            I0 = 1/(self.sigma * np.sqrt(2*np.pi)) / 10
            return I0 * np.exp(-0.5 * ((y-self.mu)**2 + z**2) / self.sigma**2)

        self.profile = gaussian
