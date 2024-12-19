import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from model_with_wegovy import pars, odde
import string
import matplotlib.transforms as mtransforms
from find_caloric_intake_in_trial import find_equilibrium_calories

initial_values_printed = False

# Simulation time periods (in days)
PRE_TREATMENT_DAYS = 1825  # 5 years
TREATMENT_DAYS = 1460  # 4 years
TOTAL_DAYS = PRE_TREATMENT_DAYS + TREATMENT_DAYS
CALORIE_REDUCTION_RAMP_DAYS = 60  # 2 months

# Initial values from default model
INITIAL_GLUCOSE = 94.1  # mg/dl
INITIAL_INSULIN = 9.6  # μU/ml
INITIAL_FFA = 404  # μmol/l
INITIAL_SI = 0.8  # ml/μU/day
INITIAL_BETA = 1009  # mg
INITIAL_SIGMA = 530  # μU/mg/day
INITIAL_INFLAMMATION = 0.056  # dimensionless
INITIAL_HEIGHT = 1.8  # m
INITIAL_AGE = 30  # years

# Update the initial values to be dictionaries based on BMI category
INITIAL_VALUES = {
    "<30": {
        "glucose": 96.21,
        "insulin": 10.43,
        "ffa": 424.02,
        "si": 0.70,
        "beta": 1010.12,
        "sigma": 541.97,
        "inflammation": 0.11,
    },
    "30-35": {
        "glucose": 102.73,
        "insulin": 12.75,
        "ffa": 481.58,
        "si": 0.50,
        "beta": 1011.75,
        "sigma": 562.44,
        "inflammation": 0.22,
    },
    "35-40": {
        "glucose": 112.93,
        "insulin": 15.19,
        "ffa": 565.77,
        "si": 0.33,
        "beta": 1010.04,
        "sigma": 540.27,
        "inflammation": 0.40,
    },
    "≥40": {
        "glucose": 122.40,
        "insulin": 15.60,
        "ffa": 642.29,
        "si": 0.26,
        "beta": 1006.71,
        "sigma": 470.42,
        "inflammation": 0.57,
    },
}


def calculate_target_weights(height_m):
    """Calculate target weights for each BMI category"""
    bmi_targets = {
        "<30": 28.0,  # Middle of normal/overweight
        "30-35": 32.5,  # Middle of class I obesity
        "35-40": 37.5,  # Middle of class II obesity
        "≥40": 42.0,  # Representative of class III obesity
    }

    # Calculate post-treatment weights based on reported reductions
    weight_reductions = {"<30": -0.1052, "30-35": -0.1179, "35-40": -0.1201, "≥40": -0.1223}

    results = {}
    for category, target_bmi in bmi_targets.items():
        initial_weight = target_bmi * (height_m**2)
        final_weight = initial_weight * (1 + weight_reductions[category])

        results[category] = {
            "initial_bmi": target_bmi,
            "initial_weight": initial_weight,
            "final_weight": final_weight,
            "final_bmi": final_weight / (height_m**2),
            "reduction": weight_reductions[category],
        }

    return results


def simulate_bmi_category(bmi_category, initial_weight):
    """Simulate a specific BMI category with pre-treatment and treatment phases"""
    global initial_values_printed

    t_span = [0, TOTAL_DAYS]
    t_eval = np.linspace(0, TOTAL_DAYS, TOTAL_DAYS + 1)  # daily points

    # Get the correct initial values based on BMI category
    category_key = "≥40" if bmi_category == "BMI ≥40" else bmi_category.replace("BMI ", "")
    initial_vals = INITIAL_VALUES[category_key]

    # Initial conditions [g, i, ffa, si, b, sigma, infl, w, height, age]
    y0 = [
        initial_vals["glucose"],
        initial_vals["insulin"],
        initial_vals["ffa"],
        initial_vals["si"],
        initial_vals["beta"],
        initial_vals["sigma"],
        initial_vals["inflammation"],
        initial_weight,
        INITIAL_HEIGHT,
        INITIAL_AGE,
    ]

    # Get weight targets from the helper function
    targets = calculate_target_weights(INITIAL_HEIGHT)
    category_key = "≥40" if bmi_category == "BMI ≥40" else bmi_category.replace("BMI ", "")
    target_data = targets[category_key]
    final_weight = target_data["final_weight"]

    # Calculate equilibrium calories for initial and treatment phases
    initial_calories, _ = find_equilibrium_calories(initial_weight, y0, simulation_years=5)
    treatment_calories, _ = find_equilibrium_calories(final_weight, y0, simulation_years=4)

    # Set up parameters
    local_pars = pars.copy()

    def custom_odde(t, y, pars_npa):
        pars_dict = dict(zip(pars.keys(), pars_npa))

        if t < PRE_TREATMENT_DAYS:  # Pre-treatment phase
            pars_dict["intake_i"] = initial_calories
        else:  # Treatment phase
            days_in_treatment = t - PRE_TREATMENT_DAYS
            if days_in_treatment <= CALORIE_REDUCTION_RAMP_DAYS:
                ramp_factor = days_in_treatment / CALORIE_REDUCTION_RAMP_DAYS
                calorie_reduction = (initial_calories - treatment_calories) * ramp_factor
                pars_dict["intake_i"] = initial_calories - calorie_reduction
            else:
                pars_dict["intake_i"] = treatment_calories

        return odde(t, y, np.array(list(pars_dict.values())), pre_treatment_years=5, treatment_years=4)

    # Run simulation
    sol = solve_ivp(custom_odde, t_span, y0, method="LSODA", t_eval=t_eval, args=(np.array(list(local_pars.values())),))

    # Store weight before BMI conversion
    weights = sol.y[7].copy()

    # Convert weight to BMI using INITIAL_HEIGHT
    sol.y[7] = sol.y[7] / (INITIAL_HEIGHT**2)

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

    # BMI categories and their initial weights
    scenarios = {
        "BMI <30": {"initial_weight": 90.72},  # BMI 28 at height 1.8m
        "BMI 30-35": {"initial_weight": 105.3},  # BMI 32.5
        "BMI 35-40": {"initial_weight": 121.5},  # BMI 37.5
        "BMI ≥40": {"initial_weight": 136.08},  # BMI 42
    }

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    # Run simulations and plot results
    for (label, scenario), color in zip(scenarios.items(), colors):
        sol = simulate_bmi_category(label, scenario["initial_weight"])
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
