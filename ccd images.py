import numpy as np
import matplotlib.pyplot as plt

dist = np.linspace(0, 20, len(d))
p_846 = np.load("CCD_simulation_ratefile_0.00846_ms_0.038_mW.npy")
p_1308 = np.load("CCD_simulation_ratefile_0.01308_ms_0.038_mW.npy")

v_f_846 = 0.1
v_f_1308 = 2.25

plt.plot(dist, p_846, drawstyle='steps-mid', label="8.46 ms")
plt.plot(dist, p_1308, drawstyle="steps-mid", label="13.08 ms")
plt.legend()