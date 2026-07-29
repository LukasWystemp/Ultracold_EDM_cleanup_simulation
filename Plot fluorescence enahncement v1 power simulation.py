import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib import rc
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit

v1 = np.array([10, 30, 44, 100])
mu = np.array([27, 37.2, 36.4, 49.1])
mu /= 14


port_a_power = np.array([6, 11, 21.7, 30, 34, 38, 44])
r_measured_8_46 = np.array([1.2, 1.3, 1.45, 1.45, 1.6, 1.6, 1.65])
r_measured_13_09 = np.array([1.3, 1.4, 1.7, 1.75, 1.9, 1.9, 2])
r_measured_17_7 = np.array([1.45, 1.65, 1.95, 2.14, 2.3, 2.3, 2.4])
r_measured_22_32 = np.array([1.55, 1.8, 2.25, 2.45, 2.6, 2.6, 2.73])
r_measured_26_94 = np.array([1.66, 1.95, 2.45, 2.7, 2.9, 2.95, 3.05])
r_measured_31_56 = np.array([1.85, 2.2, 2.75, 3.1, 3.25, 3.3, 3.5])



power_sim_31_56 = np.array([15, 30])
val_0_31_56 = 13.4
vals_31_56 = np.array([40.3, 51.9])

val_0_8_46 = 11.6
power_sim_8_46 = np.array([30])
vals_8_46 = np.array([19.3])

power_sim_22_32 = np.array([44])
val_0_22_32 = 15.2
vals_22_32 = np.array([46.6])

r_31 = vals_31_56 / val_0_31_56
r_8 = vals_8_46 / val_0_8_46
r_22 = vals_22_32 / val_0_22_32


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

ax.plot(port_a_power, r_measured_8_46, label="Measured, 8.46ms", color="tab:purple", marker="x")
ax.plot(port_a_power, r_measured_13_09, label="Measured, 13.08ms", color="tab:blue", marker="x")
ax.plot(port_a_power, r_measured_17_7, label="Measured, 17.7ms", color="tab:cyan", marker="x")
ax.plot(port_a_power, r_measured_22_32, label="Measured, 22.32ms", color="tab:green", marker="x")
ax.plot(port_a_power, r_measured_26_94, label="Measured, 26.94ms", color="tab:orange", marker="x")
ax.plot(port_a_power, r_measured_31_56, label="Measured, 31.56ms", color="tab:red", marker="x")

ax.scatter(power_sim_31_56, r_31, label="Simulation, 31.56ms", color="tab:red", marker="o", zorder=5)
ax.scatter(power_sim_8_46, r_8, label="Simulation, 8.46ms", color="tab:purple", marker="o", zorder=5)
ax.scatter(power_sim_22_32, r_22, label="Simulation, 22.32ms", color="tab:green", marker="o", zorder=5)

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

ax.legend(fontsize=8)
ax.set_title("CCD A Ratio from Det A V1 On/Off with simulation")



# ---------------- Plot 2 ----------------
bin_time = np.array([8.46, 13.08, 17.7, 22.32, 26.94, 31.56])
r_measured_6 = np.array([1.2, 1.3, 1.5, 1.6, 1.7, 1.85])
r_measured_11 = np.array([1.3, 1.4, 1.7, 1.85, 2, 2.2])
r_measured_21_7 = np.array([1.45, 1.65, 2, 2.3, 2.45, 2.7])
r_measured_30 = np.array([1.45, 1.75, 2.1, 2.45, 2.7, 3.05])
r_measured_34 = np.array([1.55, 1.9, 2.25, 2.7, 2.9, 3.25])
r_measured_38 = np.array([1.55, 1.9, 2.25, 2.72, 2.95, 3.3])
r_measured_44 = np.array([1.62, 1.95, 2.4, 2.8, 3.1, 3.4])


power_sim_31_56 = np.array([30])
val_0_31_56 = 13.4
vals_31_56 = np.array([51.9])

val_0_8_46 = 11.6
power_sim_8_46 = np.array([30])
vals_8_46 = np.array([19.3])




sim_bins = np.array([8.46, 22.32, 31.56])



powers_30_w = np.array([19.3/val_0_8_46, np.nan, 51.9/val_0_31_56])
powers_14_w = np.array([np.nan, np.nan, 40.3/val_0_31_56])
powers_44_w = np.array([np.nan, 46.6/val_0_22_32, np.nan])

datasets = [
    (r_measured_6, "6mW", "tab:red"),
    (r_measured_11, "11mW", "tab:orange"),
    (r_measured_21_7, "21.7mW", "tab:olive"),
    (r_measured_30, "30mW", "tab:green"),
    (r_measured_34, "34mW", "tab:cyan"),
    (r_measured_38, "38mW", "tab:blue"),
    (r_measured_44, "44mW", "tab:purple"),
]

fig, ax = plt.subplots()

ax.scatter(sim_bins, powers_14_w, label="Simulation, 15mW", color="tab:orange", marker="v", zorder=5, s=50)
ax.scatter(sim_bins, powers_30_w, label="Simulation, 30mW", color="tab:green", marker="v", zorder=5, s=50)
ax.scatter(sim_bins, powers_44_w, label="Simulation, 44mW", color="tab:purple", marker="v", zorder=5, s=50)

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

ax.legend(fontsize=8)
ax.set_title("CCD A Ratio from Det A V1 On/Off with simulation")

plt.show()