import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib import rc
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
import pandas as pd

v1 = np.array([10, 30, 44, 100])
mu = np.array([27, 37.2, 36.4, 49.1])
mu /= 14



power = [6, 11, 21.7, 30, 34, 38, 44]
t = [8.46, 13.08, 17.7, 22.32, 26.94, 31.56]
bin_time = t
r_measured = pd.DataFrame(np.nan, index=t, columns=power)
r_measured.loc[8.46] = [1.2, 1.3, 1.45, 1.45, 1.6, 1.6, 1.65]
r_measured.loc[13.08] = [1.3, 1.4, 1.7, 1.75, 1.9, 1.9, 2]
r_measured.loc[17.7] = [1.45, 1.65, 1.95, 2.14, 2.3, 2.3, 2.4]
r_measured.loc[22.32] = [1.55, 1.8, 2.25, 2.45, 2.6, 2.6, 2.73]
r_measured.loc[26.94] = [1.66, 1.95, 2.45, 2.7, 2.9, 2.95, 3.05]
r_measured.loc[31.56] = [1.85, 2.2, 2.75, 3.1, 3.25, 3.3, 3.5]


sim = pd.DataFrame(np.nan, index=t, columns=power)
sim.loc[:,0] = np.nan


measurements_100 = [{'P-V1': 0.0, 'bin': 0.00846, 'Np': np.float64(6.912)},
{'P-V1': 0.0, 'bin': 0.0177, 'Np': np.float64(10.746)},
{'P-V1': 0.0, 'bin': 0.02232, 'Np': np.float64(11.022)},
{'P-V1': 0.044, 'bin': 0.00846, 'Np': np.float64(9.222)},
{'P-V1': 0.0217, 'bin': 0.00846, 'Np': np.float64(9.042)},
{'P-V1': 0.038, 'bin': 0.00846, 'Np': np.float64(9.796)},
{'P-V1': 0.006, 'bin': 0.00846, 'Np': np.float64(7.878)},
{'P-V1': 0.0, 'bin': 0.03156, 'Np': np.float64(13.718)},
{'P-V1': 0.044, 'bin': 0.0177, 'Np': np.float64(22.344)},
{'P-V1': 0.0217, 'bin': 0.0177, 'Np': np.float64(19.33)},
{'P-V1': 0.006, 'bin': 0.0177, 'Np': np.float64(15.63)},
{'P-V1': 0.038, 'bin': 0.0177, 'Np': np.float64(22.704)},
{'P-V1': 0.038, 'bin': 0.02232, 'Np': np.float64(28.128)},
{'P-V1': 0.044, 'bin': 0.02232, 'Np': np.float64(28.506)},
{'P-V1': 0.006, 'bin': 0.02232, 'Np': np.float64(18.054)},
{'P-V1': 0.0217, 'bin': 0.02232, 'Np': np.float64(24.744)},
{'P-V1': 0.038, 'bin': 0.03156, 'Np': np.float64(40.052)},
{'P-V1': 0.044, 'bin': 0.03156, 'Np': np.float64(42.67)},
{'P-V1': 0.0217, 'bin': 0.03156, 'Np': np.float64(37.032)},
{'P-V1': 0.006, 'bin': 0.03156, 'Np': np.float64(23.506)}]


measurements_200 = [{'P-V1': 0.0, 'bin': 0.00846, 'Np': np.float64(10.256)},
{'P-V1': 0.0, 'bin': 0.0177, 'Np': np.float64(13.952)},
{'P-V1': 0.0, 'bin': 0.02232, 'Np': np.float64(13.014)},
{'P-V1': 0.0, 'bin': 0.03156, 'Np': np.float64(14.506)},
{'P-V1': 0.006, 'bin': 0.00846, 'Np': np.float64(11.532)},
{'P-V1': 0.044, 'bin': 0.00846, 'Np': np.float64(14.394)},
{'P-V1': 0.038, 'bin': 0.00846, 'Np': np.float64(13.96)},
{'P-V1': 0.0217, 'bin': 0.00846, 'Np': np.float64(13.346)},
{'P-V1': 0.044, 'bin': 0.0177, 'Np': np.float64(28.304)},
{'P-V1': 0.006, 'bin': 0.0177, 'Np': np.float64(18.738)},
{'P-V1': 0.0217, 'bin': 0.0177, 'Np': np.float64(25.306)},
{'P-V1': 0.038, 'bin': 0.0177, 'Np': np.float64(29.006)},
{'P-V1': 0.0217, 'bin': 0.02232, 'Np': np.float64(29.658)},
{'P-V1': 0.044, 'bin': 0.02232, 'Np': np.float64(34.532)},
{'P-V1': 0.038, 'bin': 0.02232, 'Np': np.float64(33.874)},
{'P-V1': 0.006, 'bin': 0.02232, 'Np': np.float64(21.83)},
{'P-V1': 0.038, 'bin': 0.03156, 'Np': np.float64(45.986)},
{'P-V1': 0.044, 'bin': 0.03156, 'Np': np.float64(47.986)},
{'P-V1': 0.0217, 'bin': 0.03156, 'Np': np.float64(41.082)},
{'P-V1': 0.006, 'bin': 0.03156, 'Np': np.float64(28.446)}]

# Set values here: sim.loc[tb, power]
for m in measurements_200:
    p = m["P-V1"]
    p *= 1e3
    b = m["bin"]
    b *=1e3
    N = m["Np"]
    sim.loc[b, p] = N



print(sim)
sim_r = sim
for i in range(len(sim.iloc[:,0])):
    sim_r.iloc[i,0:-1] = sim_r.iloc[i,0:-1] / sim_r.iloc[i,-1]
sim_r.drop(columns=0, inplace=True)

def sqrt_model(x, a, b, c):
    return a * np.sqrt(x + b) + c


def linear_model(x, m, c):
    return m * x + c


rc('font', **{'family': 'serif', 'serif': ['Times']})
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})

# ---------------- Plot 1 ----------------
fig, ax = plt.subplots()

ax.plot(power, r_measured.loc[8.46], label="Measured, 8.46ms", color="tab:purple", marker="x")
ax.plot(power, r_measured.loc[13.08], label="Measured, 13.08ms", color="tab:blue", marker="x")
ax.plot(power, r_measured.loc[17.7], label="Measured, 17.7ms", color="tab:cyan", marker="x")
ax.plot(power, r_measured.loc[22.32], label="Measured, 22.32ms", color="tab:green", marker="x")
ax.plot(power, r_measured.loc[26.94] , label="Measured, 26.94ms", color="tab:orange", marker="x")
ax.plot(power, r_measured.loc[31.56], label="Measured, 31.56ms", color="tab:red", marker="x")

ax.scatter(power, sim_r.loc[8.46], label="Simulation, 8.46ms", color="tab:purple", marker="o", edgecolors="black", linewidths=0.5)
ax.scatter(power, sim_r.loc[13.08], label="Simulation, 13.08ms", color="tab:blue", marker="o", edgecolors="black", linewidths=0.5)
ax.scatter(power, sim_r.loc[17.7], label="Simulation, 17.7ms", color="tab:cyan", marker="o", edgecolors="black", linewidths=0.5)
ax.scatter(power, sim_r.loc[22.32], label="Simulation, 22.32ms", color="tab:green", marker="o", edgecolors="black", linewidths=0.5)
ax.scatter(power, sim_r.loc[26.94] , label="Simulation, 26.94ms", color="tab:orange", marker="o", edgecolors="black", linewidths=0.5)
ax.scatter(power, sim_r.loc[31.56], label="Simulation, 31.56ms", color="tab:red", marker="o",edgecolors="black", linewidths=0.5)


ax.set_ylim(1, 5)
ax.set_xlim(5, 45)
ax.set_ylabel("Ratio")
ax.set_xlabel("V1 Port A Power, mW")

ax.xaxis.set_major_locator(MultipleLocator(10))
ax.yaxis.set_major_locator(MultipleLocator(1))
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.yaxis.set_minor_locator(MultipleLocator(0.1))

ax.tick_params(which='minor', length=4, width=0.8)
ax.tick_params(which='major', length=7, width=1.2)

ax.legend(fontsize=8, ncol=2)
ax.set_title("CCD A Ratio from Det A V1 On/Off with simulation")



# ---------------- Plot 2 ----------------


datasets = [
    (r_measured.loc[:,6], "6 mW", "tab:red"),
    (r_measured.loc[:,11], "11 mW", "tab:orange"),
    (r_measured.loc[:,21.7], "21.7 mW", "tab:olive"),
    (r_measured.loc[:,30], "30 mW", "tab:green"),
    (r_measured.loc[:,34], "34 mW", "tab:cyan"),
    (r_measured.loc[:,38], "38 mW", "tab:blue"),
    (r_measured.loc[:,44], "44 mW", "tab:purple"),
]

fig, ax = plt.subplots()


ax.scatter(t, sim_r.loc[:,6], label="Simulation, 6 mW", color="tab:red", marker="v", zorder=5, s=50, edgecolors="black", linewidths=0.5)
ax.scatter(t, sim_r.loc[:,11], label="Simulation, 11 mW", color="tab:orange", marker="v", zorder=5, s=50, edgecolors="black", linewidths=0.5)
ax.scatter(t, sim_r.loc[:,21.7], label="Simulation, 21.7 mW", color="tab:olive", marker="v", zorder=5, s=50, edgecolors="black", linewidths=0.5)
ax.scatter(t, sim_r.loc[:,30], label="Simulation, 30 mW", color="tab:green", marker="v", zorder=5, s=50, edgecolors="black", linewidths=0.5)
ax.scatter(t, sim_r.loc[:,34], label="Simulation, 34 mW", color="tab:cyan", marker="v", zorder=5, s=50, edgecolors="black", linewidths=0.5)
ax.scatter(t, sim_r.loc[:,38], label="Simulation, 38 mW", color="tab:blue", marker="v", zorder=5, s=50, edgecolors="black", linewidths=0.5)
ax.scatter(t, sim_r.loc[:,44], label="Simulation, 44 mW", color="tab:purple", marker="v", zorder=5, s=50, edgecolors="black", linewidths=0.5)

x_fit = np.linspace(5, 34, 100)
for data, label, color in datasets:
    popt, pcov = curve_fit(linear_model, bin_time, data)
    m, c = popt
    ax.scatter(bin_time, data, label=f"Measured, {label}", color=color, marker="o", s=10)
    ax.plot(x_fit, linear_model(x_fit, m, c), color=color)


ax.set_ylim(1, 4)
ax.set_xlim(5, 34)
ax.set_ylabel("Ratio")
ax.set_xlabel("Bin Time, ms")

ax.xaxis.set_major_locator(MultipleLocator(5))
ax.yaxis.set_major_locator(MultipleLocator(1))
ax.xaxis.set_minor_locator(MultipleLocator(0.5))
ax.yaxis.set_minor_locator(MultipleLocator(0.1))

ax.tick_params(which='minor', length=4, width=0.8)
ax.tick_params(which='major', length=7, width=1.2)

ax.legend(fontsize=8, ncol=2)
ax.set_title("CCD A Ratio from Det A V1 On/Off with simulation")

plt.show()