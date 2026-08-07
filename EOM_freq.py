import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib import rc
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
import pandas as pd



power = [6, 11, 21.7, 30, 34, 38, 44, 0]
t = [8.46, 13.08, 17.7, 22.32, 26.94, 31.56]
f = np.array([0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10])*1e6



sim_8_46 = pd.DataFrame(np.nan, index=f, columns=power)
sim_8_46.loc[:,0] = np.nan
sim_31_56 = sim_8_46.copy()

print(sim_8_46)

sim_8_46.loc[:, 0]    = [5.362, 5.996,  7.668, 10.422, 12.094, 11.134, 12.038, 12.344]
sim_8_46.loc[:, 6]    = [5.752, 6.264,  9.068, 13.070, 13.918, 13.036, 14.780, 14.550]
sim_8_46.loc[:, 21.7] = [6.090, 6.900,  9.936, 15.062, 16.988, 17.082, 18.312, 18.174]
sim_8_46.loc[:, 44]   = [6.658, 7.162, 10.350, 16.508, 18.612, 18.050, 22.158, 20.360]
 
# bin = 31.56 ms  (0.03156 s)
sim_31_56.loc[:, 0]    = [11.536, 11.176, 13.476, 15.012, 15.434, 14.674, 14.974, 14.740]
sim_31_56.loc[:, 6]    = [15.352, 19.378, 21.882, 28.678, 30.932, 30.310, 32.270, 29.852]
sim_31_56.loc[:, 21.7] = [18.180, 22.912, 30.670, 41.382, 45.292, 47.386, 52.580, 48.504]
sim_31_56.loc[:, 44]   = [19.952, 24.546, 33.162, 47.260, 54.922, 55.774, 66.404, 62.904]

print(sim_8_46)
print(sim_31_56)

"""
for _, j in enumerate([sim_8_46, sim_17_7, sim_31_56]):
    for i in range(len(j.iloc[:,0])):
        j.iloc[i,0:-1] /= j.iloc[i,-1]
    j.drop(columns=0,inplace=True)
"""


rc('font', **{'family': 'serif', 'serif': ['Times']})
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})

power_colours = np.array([
    ["6 mW", "tab:red"],
    ["11 mW", "tab:orange"],
    ["21.7 mW", "tab:olive"],
    ["30 mW", "tab:green"],
    ["34 mW", "tab:cyan"],
    ["38 mW", "tab:blue"],
    ["44 mW", "tab:purple"],
    ["0 mW", "black"],
])

tbs = ["8.46", "31.56"]
for tbi, j in enumerate([sim_8_46, sim_31_56]):
    fig, ax = plt.subplots()


    for i in range(len(j.iloc[0,:])):
        if not np.isnan(j.iloc[0,i]):
            ax.scatter(j.index, j.iloc[:,i], label=power_colours[i,0], color=power_colours[i,1], marker="v", zorder=5, s=50, edgecolors="black", linewidths=0.5)
            
    ax.set_ylabel("Photons per Molecule over transit")
    ax.set_xlabel("Polarisation Modulation (Hz)")


    #ax.xaxis.set_major_locator(MultipleLocator(0.1))
    #ax.yaxis.set_major_locator(MultipleLocator(0.2))
    #ax.xaxis.set_minor_locator(MultipleLocator(0.01))
    #ax.yaxis.set_minor_locator(MultipleLocator(0.05))

    ax.tick_params(which='minor', length=4, width=0.8)
    ax.tick_params(which='major', length=7, width=1.2)
    #ax.set_xscale("log")
    ax.axvline(900e3, color="grey", ls="--", lw=1, zorder=1, label="900 kHz")

    ax.legend(fontsize=8)
    ax.set_title(f"CCD A Simulated Scattering from Det A V1 | {tbs[tbi]} ms")
    plt.plot()
    fig.savefig(f"CCD_A_Simulated_Scattering_from_Det_A_V1_|_{tbs[tbi]}_ms_over_EOM_frequency.png", dpi=300)
