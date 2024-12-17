# figure2.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp as sivp
from scipy.interpolate import interp1d
import string
import matplotlib.transforms as mtransforms
from model import *  # Importing the model definition from model.py

# Initial values for the variables
y0 = [0] * 8
y0[0] = 94.1  # Initial glucose level (mg/dl)
y0[1] = 9.6  # Initial insulin level (uU/ml)
y0[2] = 404  # Initial free fatty acids (FFA) (umol/l)
y0[3] = 0.8  # Initial insulin sensitivity (Si) (ml/uU/day)
y0[4] = 1009  # Initial beta cell mass (mg)
y0[5] = 530  # Initial beta cell function (sigma) (uU/mg/day)
y0[6] = 0.056  # Initial systemic inflammation (dimensionless)
y0[7] = 81  # Initial body weight (kg)

# Time span for the simulation, from day 0 to day 1800
tspan = [0, 3600]

# Different increments in dietary intake to simulate various scenarios
incivals = [0.3, 0.55, 0.75]

# Styles for each scenario
lss = [":", "-", "--"]
cols = ["royalblue", "limegreen", "magenta"]
incilab = [round(i * 100) for i in incivals]

# Variable names for graph labeling, using LaTeX for formatting
vn = [
    r"G ($mg/dl$)",
    r"I ($\mu U/ml$)",
    r"FFA ($\mu mol/l$)",
    r"S$_i$ ($ml/ \mu U/day$)",
    r"$\beta$ ($mg$)",
    r"$\sigma$ ($\mu U/mg/day$)",
    r"$\theta$",
    "BMI ($kg/m^2$)",
]

# Create a figure with a grid of 3x3 subplots
fig5 = plt.figure(12, tight_layout=False)
axs = fig5.subplots(3, 3)
axs = axs.ravel()  # Flatten the array of axes for easier indexing
letter = string.ascii_uppercase
trans = mtransforms.ScaledTranslation(-20 / 72, 7 / 72, fig5.dpi_scale_trans)

# Labeling and setting up each subplot
for i in range(len(axs) - 1):
    axs[i].text(
        -0.1,
        1.05,
        letter[i],
        transform=axs[i].transAxes,
        fontsize="medium",
        weight="bold",
        va="bottom",
        fontfamily="arial",
    )
    axs[i].tick_params(axis="both", which="major", labelsize=10)

    if i == 0:
        axs[i].set_ylim([90, 150])
    if i == 1:
        axs[i].set_ylim([9, 17])
    if i == 2:
        axs[i].set_ylim([400, 720])
    if i == 3:
        axs[i].set_ylim([0.2, 0.9])
    if i == 4:
        axs[i].set_ylim([720, 1250])
    if i == 5:
        axs[i].set_ylim([350, 570])
    if i == 6:
        axs[i].set_ylim([0, 0.7])
    if i == 7:
        axs[i].set_ylim([20, 45])

for i1, inci in enumerate(incivals):

    pars["inc_i1"] = inci
    pars["it1"] = 150
    pars_npa = np.array(list(pars.values()))

    sol = sivp(odde, tspan, y0, method="LSODA", args=[pars_npa])
    sol.y[7] = sol.y[7] / (pars["height"] ** 2)
    t_sim = sol.t / 30

    # Plotting Disease progression index (DPI=si*b*sigma/(si0*b0*sigma0)
    dp = 1 - sol.y[3] * sol.y[4] * sol.y[5] / (sol.y[3][0] * sol.y[4][0] * sol.y[5][0])

    fdp = interp1d(t_sim, dp)
    tps = np.arange(0, 121, 12)
    axs[8].bar(tps - 3 + 3 * i1, fdp(tps), color=cols[i1], width=3)
    axs[8].set_xlabel("Time (month)", size="medium")
    axs[8].set_ylabel("DPI", size="medium", labelpad=2)
    axs[8].set_xlim([t_sim[0], t_sim[-1] + 5])
    axs[8].set_ylim([0, 1])

    old = 0

    for i, j in enumerate(sol.y):
        axs[i].plot(t_sim, j, c=cols[i1], label=f"DE$_i$={100+incilab[i1]}% BL", linestyle="-")
        axs[i].set_xlabel("Time (month)", size="medium")
        axs[i].set_ylabel(vn[i], size="medium", labelpad=2)
        axs[i].set_xlim([min(t_sim), max(t_sim)])
        if i == 0:
            axs[i].legend(fontsize="small", framealpha=0.5, ncol=3, loc=(0, 1.25))
            if old == 0:
                axs[i].hlines(
                    y=[100, 125], xmin=0, xmax=max(sol.t) / 30, linestyles=":", colors=["k", "r"], linewidth=0.8
                )
                old = 1

fig5.tight_layout()
fig5.subplots_adjust(top=0.88, bottom=0.09, left=0.08, right=0.98, hspace=0.6, wspace=0.45)
fig5.set_size_inches([8.5, 5.7])

plt.show()
