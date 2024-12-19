import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from model_with_wegovy import pars, odde
import string
import matplotlib.transforms as mtransforms

initial_values_printed = False


def calculate_maintenance_calories(weight_kg, height_m, age=30, sex=1, activity_factor=1.2):
    """Calculate maintenance calories using Mifflin-St Jeor equation"""
    height_cm = height_m * 100
    if sex == 1:  # male
        bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) + 5
    else:  # female
        bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) - 161
    return bmr * activity_factor


def simulate_bmi_category(bmi_category, pre_calories, treatment_calories, initial_weight=81):
    """Simulate a specific BMI category with pre-treatment and treatment phases"""
    global initial_values_printed

    t_span = [0, 2555]  # 7 years total (5 years pre + 2 years treatment)
    t_eval = np.linspace(0, 2555, 2556)  # daily points

    # Initial conditions [g, i, ffa, si, b, sigma, infl, w]
    y0 = [94.1, 9.6, 404, 0.8, 1009, 530, 0.056, initial_weight]

    # Set up parameters
    local_pars = pars.copy()

    def custom_odde(t, y, pars_npa):
        pars_dict = dict(zip(pars.keys(), pars_npa))

        if t < 1825:  # First 5 years - weight gain phase
            pars_dict["intake_i"] = pre_calories
        else:  # Treatment phase
            days_in_treatment = t - 1825
            if days_in_treatment <= 60:  # 2-month ramp up
                ramp_factor = days_in_treatment / 60  # Linear ramp from 0 to 1
                calorie_reduction = (pre_calories - treatment_calories) * ramp_factor
                pars_dict["intake_i"] = pre_calories - calorie_reduction
            else:  # Full treatment
                pars_dict["intake_i"] = treatment_calories

        return odde(t, y, np.array(list(pars_dict.values())), pre_treatment_years=5, treatment_years=2)

    # Run simulation
    sol = solve_ivp(custom_odde, t_span, y0, method="LSODA", t_eval=t_eval, args=(np.array(list(local_pars.values())),))

    # Store weight before BMI conversion
    weights = sol.y[7].copy()

    # Convert weight to BMI
    sol.y[7] = sol.y[7] / (pars["height"] ** 2)

    # Print values at key timepoints
    variables = ["Glucose", "Insulin", "FFA", "Si", "Beta", "Sigma", "Inflammation", "BMI", "Weight (kg)"]

    # Initial values (t=0) - only print once
    if not initial_values_printed:
        print("\nInitial values (all groups):")
        for var, val in zip(variables[:-1], [y[0] for y in sol.y]):
            print(f"{var:12}: {val:.2f}")
        print(f"{'Weight (kg)':12}: {weights[0]:.2f}")
        initial_values_printed = True

    # Print the bmi category
    print(f"\n=== {bmi_category} ===")

    # Post weight gain (t=1825, 5 years)
    print("\nPost weight gain (5 years):")
    idx_5y = 1825
    for var, vals in zip(variables[:-1], sol.y):
        print(f"{var:12}: {vals[idx_5y]:.2f}")
    print(f"{'Weight (kg)':12}: {weights[idx_5y]:.2f}")

    # Final values (t=2555, 7 years)
    print("\nPost treatment (7 years):")
    idx_7y = -1
    for var, vals in zip(variables[:-1], sol.y):
        print(f"{var:12}: {vals[idx_7y]:.2f}")
    print(f"{'Weight (kg)':12}: {weights[idx_7y]:.2f}")

    # Calculate DPI at final timepoint
    dpi = 1 - (sol.y[3][idx_7y] * sol.y[4][idx_7y] * sol.y[5][idx_7y]) / (sol.y[3][0] * sol.y[4][0] * sol.y[5][0])
    print(f"{'DPI':12}: {dpi:.2f}")

    return sol


def plot_trial_results():
    """Plot biomarkers for all BMI categories"""
    # Variable names with LaTeX formatting
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

    # Create figure with matching size and layout
    fig = plt.figure(tight_layout=False)
    axs = fig.subplots(3, 3)
    axs = axs.ravel()

    # Add subplot labels (A, B, C, etc.)
    letter = string.ascii_uppercase
    trans = mtransforms.ScaledTranslation(-20 / 72, 7 / 72, fig.dpi_scale_trans)

    # Set y-limits and labels for each subplot
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

        # Match y-limits from figure2
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

    # BMI categories and their caloric patterns from outline.txt
    scenarios = {
        "BMI <30": {"pre_calories": 2797, "treatment_calories": 2797 * 0.8948},
        "BMI 30-35": {"pre_calories": 3266, "treatment_calories": 3266 * 0.8821},
        "BMI 35-40": {"pre_calories": 3754, "treatment_calories": 3754 * 0.8799},
        "BMI ≥40": {"pre_calories": 4203, "treatment_calories": 4203 * 0.8777},
    }

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    # Run simulations and plot results
    for (label, scenario), color in zip(scenarios.items(), colors):
        sol = simulate_bmi_category(label, scenario["pre_calories"], scenario["treatment_calories"])
        years = sol.t / 365

        # Plot each variable with consistent styling
        for i in range(8):
            axs[i].plot(years, sol.y[i], c=color, label=label)
            axs[i].set_xlabel("Time (years)", size="medium")
            axs[i].set_ylabel(vn[i], size="medium", labelpad=2)

        # DPI calculation and plotting
        dpi = 1 - (sol.y[3] * sol.y[4] * sol.y[5]) / (sol.y[3][0] * sol.y[4][0] * sol.y[5][0])
        axs[8].plot(years, dpi, c=color, label=label)

    # Configure DPI subplot
    axs[8].set_xlabel("Time (years)", size="medium")
    axs[8].set_ylabel("DPI", size="medium", labelpad=2)
    axs[8].set_ylim([0, 1])

    # Add legend to first subplot with matching style
    axs[0].legend(fontsize="small", framealpha=0.5, ncol=2, loc=(0, 1.25))

    # Add reference lines to glucose plot
    axs[0].hlines(y=[100, 125], xmin=0, xmax=max(years), linestyles=":", colors=["k", "r"], linewidth=0.8)

    # Match figure size and layout adjustments
    fig.tight_layout()
    fig.subplots_adjust(top=0.88, bottom=0.09, left=0.08, right=0.98, hspace=0.6, wspace=0.45)
    fig.set_size_inches([8.5, 5.7])

    plt.show()


if __name__ == "__main__":
    plot_trial_results()
