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
B = [0, 0.2, 0.5, 1]


sim_8_46 = pd.DataFrame(np.nan, index=B, columns=power)
sim_8_46.loc[:,0] = np.nan
sim_17_7 = sim_8_46.copy()
sim_22_32 = sim_8_46.copy()
sim_31_56 = sim_8_46.copy()

print(sim_8_46)
data = [{'P-V1': 0.0, 'bin': 0.00846, 'B': 0, 'Np': np.float64(6.98)},
{'P-V1': 0.0, 'bin': 0.00846, 'B': 0.2, 'Np': np.float64(7.228)},
{'P-V1': 0.0, 'bin': 0.00846, 'B': 0.5, 'Np': np.float64(5.3)},
{'P-V1': 0.0, 'bin': 0.00846, 'B': 1, 'Np': np.float64(3.464)},
{'P-V1': 0.0, 'bin': 0.0177, 'B': 0.2, 'Np': np.float64(11.492)},
{'P-V1': 0.0, 'bin': 0.0177, 'B': 0, 'Np': np.float64(8.782)},
{'P-V1': 0.0, 'bin': 0.02232, 'B': 0.2, 'Np': np.float64(13.218)},
{'P-V1': 0.0, 'bin': 0.0177, 'B': 0.5, 'Np': np.float64(8.894)},
{'P-V1': 0.0, 'bin': 0.03156, 'B': 0.2, 'Np': np.float64(12.748)},
{'P-V1': 0.044, 'bin': 0.00846, 'B': 0.2, 'Np': np.float64(10.122)},
{'P-V1': 0.038, 'bin': 0.00846, 'B': 0, 'Np': np.float64(9.684)},
{'P-V1': 0.0217, 'bin': 0.00846, 'B': 0.2, 'Np': np.float64(9.422)},
{'P-V1': 0.038, 'bin': 0.00846, 'B': 0.2, 'Np': np.float64(9.896)},
{'P-V1': 0.044, 'bin': 0.00846, 'B': 0.5, 'Np': np.float64(6.884)},
{'P-V1': 0.006, 'bin': 0.00846, 'B': 0.2, 'Np': np.float64(8.264)},
{'P-V1': 0.006, 'bin': 0.00846, 'B': 0, 'Np': np.float64(7.448)},
{'P-V1': 0.044, 'bin': 0.00846, 'B': 0, 'Np': np.float64(9.886)},
{'P-V1': 0.0, 'bin': 0.02232, 'B': 0, 'Np': np.float64(9.01)},
{'P-V1': 0.0217, 'bin': 0.00846, 'B': 0, 'Np': np.float64(9.096)},
{'P-V1': 0.0217, 'bin': 0.00846, 'B': 1, 'Np': np.float64(4.074)},
{'P-V1': 0.0217, 'bin': 0.00846, 'B': 0.5, 'Np': np.float64(6.22)},
{'P-V1': 0.038, 'bin': 0.00846, 'B': 0.5, 'Np': np.float64(6.956)},
{'P-V1': 0.006, 'bin': 0.00846, 'B': 0.5, 'Np': np.float64(5.966)},
{'P-V1': 0.044, 'bin': 0.00846, 'B': 1, 'Np': np.float64(4.236)},
{'P-V1': 0.038, 'bin': 0.00846, 'B': 1, 'Np': np.float64(4.492)},
{'P-V1': 0.006, 'bin': 0.00846, 'B': 1, 'Np': np.float64(3.798)},
{'P-V1': 0.0, 'bin': 0.02232, 'B': 0.5, 'Np': np.float64(10.34)},
{'P-V1': 0.0, 'bin': 0.0177, 'B': 1, 'Np': np.float64(5.854)},
{'P-V1': 0.0, 'bin': 0.02232, 'B': 1, 'Np': np.float64(7.324)},
{'P-V1': 0.0, 'bin': 0.03156, 'B': 0.5, 'Np': np.float64(11.526)},
{'P-V1': 0.0, 'bin': 0.03156, 'B': 0, 'Np': np.float64(10.032)},
{'P-V1': 0.0, 'bin': 0.03156, 'B': 1, 'Np': np.float64(9.056)},
{'P-V1': 0.0217, 'bin': 0.0177, 'B': 0, 'Np': np.float64(16.748)},
{'P-V1': 0.0217, 'bin': 0.0177, 'B': 0.2, 'Np': np.float64(20.788)},
{'P-V1': 0.038, 'bin': 0.0177, 'B': 0.2, 'Np': np.float64(23.25)},
{'P-V1': 0.044, 'bin': 0.0177, 'B': 0, 'Np': np.float64(17.294)},
{'P-V1': 0.044, 'bin': 0.0177, 'B': 0.2, 'Np': np.float64(23.666)},
{'P-V1': 0.038, 'bin': 0.0177, 'B': 0.5, 'Np': np.float64(18.916)},
{'P-V1': 0.044, 'bin': 0.0177, 'B': 0.5, 'Np': np.float64(19.194)},
{'P-V1': 0.006, 'bin': 0.0177, 'B': 0.2, 'Np': np.float64(16.134)},
{'P-V1': 0.006, 'bin': 0.0177, 'B': 0, 'Np': np.float64(12.186)},
{'P-V1': 0.006, 'bin': 0.0177, 'B': 0.5, 'Np': np.float64(12.668)},
{'P-V1': 0.0217, 'bin': 0.0177, 'B': 0.5, 'Np': np.float64(17.316)},
{'P-V1': 0.0217, 'bin': 0.0177, 'B': 1, 'Np': np.float64(10.242)},
{'P-V1': 0.038, 'bin': 0.0177, 'B': 0, 'Np': np.float64(17.744)},
{'P-V1': 0.044, 'bin': 0.0177, 'B': 1, 'Np': np.float64(11.308)},
{'P-V1': 0.038, 'bin': 0.0177, 'B': 1, 'Np': np.float64(10.814)},
{'P-V1': 0.006, 'bin': 0.0177, 'B': 1, 'Np': np.float64(8.092)},
{'P-V1': 0.044, 'bin': 0.02232, 'B': 0.2, 'Np': np.float64(31.732)},
{'P-V1': 0.038, 'bin': 0.02232, 'B': 0.2, 'Np': np.float64(29.72)},
{'P-V1': 0.0217, 'bin': 0.02232, 'B': 0.2, 'Np': np.float64(25.234)},
{'P-V1': 0.038, 'bin': 0.02232, 'B': 0.5, 'Np': np.float64(24.596)},
{'P-V1': 0.006, 'bin': 0.02232, 'B': 0.2, 'Np': np.float64(20.084)},
{'P-V1': 0.006, 'bin': 0.02232, 'B': 0.5, 'Np': np.float64(15.31)},
{'P-V1': 0.0217, 'bin': 0.02232, 'B': 1, 'Np': np.float64(13.542)},
{'P-V1': 0.044, 'bin': 0.02232, 'B': 1, 'Np': np.float64(16.024)},
{'P-V1': 0.038, 'bin': 0.02232, 'B': 1, 'Np': np.float64(15.328)},
{'P-V1': 0.038, 'bin': 0.02232, 'B': 0, 'Np': np.float64(19.19)},
{'P-V1': 0.0217, 'bin': 0.02232, 'B': 0.5, 'Np': np.float64(21.594)},
{'P-V1': 0.044, 'bin': 0.02232, 'B': 0.5, 'Np': np.float64(25.502)},
{'P-V1': 0.0217, 'bin': 0.02232, 'B': 0, 'Np': np.float64(18.244)},
{'P-V1': 0.044, 'bin': 0.02232, 'B': 0, 'Np': np.float64(21.07)},
{'P-V1': 0.006, 'bin': 0.02232, 'B': 1, 'Np': np.float64(11.188)},
{'P-V1': 0.006, 'bin': 0.02232, 'B': 0, 'Np': np.float64(14.858)},
{'P-V1': 0.006, 'bin': 0.03156, 'B': 0.2, 'Np': np.float64(24.302)},
{'P-V1': 0.044, 'bin': 0.03156, 'B': 0.2, 'Np': np.float64(42.714)},
{'P-V1': 0.006, 'bin': 0.03156, 'B': 0.5, 'Np': np.float64(21.332)},
{'P-V1': 0.0217, 'bin': 0.03156, 'B': 0.2, 'Np': np.float64(35.098)},
{'P-V1': 0.044, 'bin': 0.03156, 'B': 0.5, 'Np': np.float64(36.27)},
{'P-V1': 0.038, 'bin': 0.03156, 'B': 0.2, 'Np': np.float64(40.356)},
{'P-V1': 0.044, 'bin': 0.03156, 'B': 1, 'Np': np.float64(26.61)},
{'P-V1': 0.0217, 'bin': 0.03156, 'B': 1, 'Np': np.float64(22.416)},
{'P-V1': 0.038, 'bin': 0.03156, 'B': 0.5, 'Np': np.float64(35.45)},
{'P-V1': 0.044, 'bin': 0.03156, 'B': 0, 'Np': np.float64(25.694)},
{'P-V1': 0.038, 'bin': 0.03156, 'B': 0, 'Np': np.float64(22.734)},
{'P-V1': 0.0217, 'bin': 0.03156, 'B': 0.5, 'Np': np.float64(30.248)},
{'P-V1': 0.006, 'bin': 0.03156, 'B': 0, 'Np': np.float64(17.702)},
{'P-V1': 0.0217, 'bin': 0.03156, 'B': 0, 'Np': np.float64(22.972)},
{'P-V1': 0.038, 'bin': 0.03156, 'B': 1, 'Np': np.float64(24.782)},
{'P-V1': 0.006, 'bin': 0.03156, 'B': 1, 'Np': np.float64(14.822)}]

for i in data:
    p, tb, B, N = i["P-V1"], i["bin"], i["B"], i["Np"]
    if tb == 0.00846:
        sim_8_46[B, p] = N
    elif tb == 0.02232:
        sim_22_32[B, p] = N
    elif tb == 0.0177:
        sim_17_7[B, p] = N
    elif tb == 0.03156:
        sim_31_56[B, p] = N
    else:
        raise KeyError(f"{tb}, {p} not found")


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
