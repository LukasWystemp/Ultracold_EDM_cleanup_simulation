import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib import rc
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
import pandas as pd




power = [6, 11, 21.7, 30, 34, 38, 44]
t = [8.46, 13.08, 17.7, 22.32, 26.94, 31.56]
B = [0.15, 0.3, 0.45, 0.6]
r_measured = pd.DataFrame(np.nan, index=t, columns=power)
r_measured.loc[8.46] = [1.2, 1.3, 1.45, 1.45, 1.6, 1.6, 1.65]
r_measured.loc[13.08] = [1.3, 1.4, 1.7, 1.75, 1.9, 1.9, 2]
r_measured.loc[17.7] = [1.45, 1.65, 1.95, 2.14, 2.3, 2.3, 2.4]
r_measured.loc[22.32] = [1.55, 1.8, 2.25, 2.45, 2.6, 2.6, 2.73]
r_measured.loc[26.94] = [1.66, 1.95, 2.45, 2.7, 2.9, 2.95, 3.05]
r_measured.loc[31.56] = [1.85, 2.2, 2.75, 3.1, 3.25, 3.3, 3.5]


sim_8_46 = pd.DataFrame(np.nan, index=B, columns=power)
sim_8_46.loc[:,0] = np.nan
sim_17_7 = sim_8_46.copy()
sim_31_56 = sim_8_46.copy()

# Set values here: sim.loc[tb, power]
sim_8_46.loc[0.15, 0] = 12.14
sim_8_46.loc[0.45, 0] = 12.92
sim_8_46.loc[0.3, 0] = 12.18
sim_8_46.loc[0.6, 0] = 11.94
sim_17_7.loc[0.15, 0] = 13.5
sim_17_7.loc[0.3, 0] = 13.35
sim_17_7.loc[0.6, 0] = 14.3
sim_17_7.loc[0.45, 0] = 13.5
sim_31_56.loc[0.3,0] = 12.99
sim_31_56.loc[0.45,0] = 13.3
sim_31_56.loc[0.6,0] = 14.5
sim_31_56.loc[0.15,0] = 15.3
sim_8_46.loc[0.6,44]=18.93
sim_8_46.loc[0.6, 21.7]=15.7
sim_8_46.loc[0.3,21.7]=15.7
sim_8_46.loc[0.45, 21.7] = 15.9
sim_8_46.loc[0.6,6]=14.5
sim_8_46.loc[0.15,6]=15.04
sim_8_46.loc[0.15,44]=18
sim_8_46.loc[0.3,6]=13.7
sim_8_46.loc[0.15,21.7]=16.9
sim_8_46.loc[0.45,44]=17.3
sim_8_46.loc[0.45,6]=12.85
sim_8_46.loc[0.3,44]=18.07
sim_17_7.loc[0.45,44]=34.3
sim_17_7.loc[0.3,44]=32.3
sim_17_7.loc[0.15,44]=35.4
sim_17_7.loc[0.6,21.7]=29.1
sim_17_7.loc[0.6,44]=34.9
sim_17_7.loc[0.6,6]=20.994
sim_17_7.loc[0.3,21.7]=29
sim_17_7.loc[0.3,6]=21.4
sim_17_7.loc[0.15,21.7]=29.1
sim_17_7.loc[0.45,21.7]=30.8
sim_17_7.loc[0.45,6]=22.7
sim_17_7.loc[0.15,6]=21.5
sim_31_56.loc[0.15,6]=21.5
sim_31_56.loc[0.15,44]=53.3
sim_31_56.loc[0.6,44]=57.1
sim_31_56.loc[0.45,44]=56.8
sim_31_56.loc[0.6,21.7]=46.4
sim_31_56.loc[0.15,21.7]=44.6
sim_31_56.loc[0.3,44]=55.2
sim_31_56.loc[0.3,21.7]=45.9
sim_31_56.loc[0.45,21.7]=47.7
sim_31_56.loc[0.15,6]=31.95
sim_31_56.loc[0.6,6]=31.9
sim_31_56.loc[0.3,6]=31.9
sim_31_56.loc[0.45,6]=31.2



for _, j in enumerate([sim_8_46, sim_17_7, sim_31_56]):
    for i in range(len(j.iloc[:,0])):
        j.iloc[i,0:-1] /= j.iloc[i,-1]
    j.drop(columns=0,inplace=True)


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
])

tbs = ["8.46", "17.7", "31.56"]
for tbi, j in enumerate([sim_8_46, sim_17_7, sim_31_56]):
    fig, ax = plt.subplots()


    for i in range(len(j.iloc[0,:])):
        if not np.isnan(j.iloc[0,i]):
            ax.scatter(j.index, j.iloc[:,i], label=power_colours[i,0], color=power_colours[i,1], marker="v", zorder=5, s=50, edgecolors="black", linewidths=0.5)
            
    ax.set_ylabel("Ratio")
    ax.set_xlabel("B-field strength (Gauss)")

  

    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.01))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))

    ax.tick_params(which='minor', length=4, width=0.8)
    ax.tick_params(which='major', length=7, width=1.2)


    ax.legend(fontsize=8)
    ax.set_title(f"CCD A Simulated Ratio from Det A V1 On/Off | {tbs[tbi]} ms")
    plt.plot()
