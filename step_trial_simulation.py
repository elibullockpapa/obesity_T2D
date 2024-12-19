import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from model_with_wegovy import pars, odde
import string
import matplotlib.transforms as mtransforms

# Simulation time periods (in days)
PRE_TREATMENT_DAYS = 0  # no pre-treatment
TREATMENT_DAYS = 730  # 2 years
TOTAL_DAYS = PRE_TREATMENT_DAYS + TREATMENT_DAYS
CALORIE_REDUCTION_RAMP_DAYS = 60  # 2 months

# Initial conditions
INITIAL_GLUCOSE = 95.4  # mg/dl
INITIAL_INSULIN = 12.6  # μU/ml
INITIAL_FFA = 400  # μmol/l
INITIAL_SI = 0.8  # ml/μU/day
INITIAL_BETA = 1009  # mg
INITIAL_SIGMA = 530  # μU/mg/day
INITIAL_INFLAMMATION = 0.056  # dimensionless
INITIAL_WEIGHT = 105.6  # kg
INITIAL_HEIGHT = 1.65  # m
INITIAL_AGE = 47.3  # years

initial_values_printed = False


def simulate_step_trial(
    pre_calories, treatment_calories, initial_weight=INITIAL_WEIGHT, height=INITIAL_HEIGHT, age=INITIAL_AGE
):
    """Simulate STEP trial with pre-treatment and treatment phases"""
    global initial_values_printed

    t_span = [0, TREATMENT_DAYS]
    t_eval = np.linspace(0, TREATMENT_DAYS, TREATMENT_DAYS + 1)  # daily points

    # Initial conditions [g, i, ffa, si, b, sigma, infl, w, height, age]
    y0 = [
        INITIAL_GLUCOSE,
        INITIAL_INSULIN,
        INITIAL_FFA,
        INITIAL_SI,
        INITIAL_BETA,
        INITIAL_SIGMA,
        INITIAL_INFLAMMATION,
        initial_weight,
        height,
        age,
    ]

    # Set up parameters
    local_pars = pars.copy()

    def custom_odde(t, y, pars_npa):
        pars_dict = dict(zip(pars.keys(), pars_npa))

        if t < PRE_TREATMENT_DAYS:  # First 10 years - pre-treatment phase
            pars_dict["intake_i"] = pre_calories
        else:  # Treatment phase
            days_in_treatment = t - PRE_TREATMENT_DAYS
            if days_in_treatment <= CALORIE_REDUCTION_RAMP_DAYS:  # 2-month ramp up
                ramp_factor = days_in_treatment / CALORIE_REDUCTION_RAMP_DAYS  # Linear ramp from 0 to 1
                calorie_reduction = (pre_calories - treatment_calories) * ramp_factor
                pars_dict["intake_i"] = pre_calories - calorie_reduction
            else:  # Full treatment
                pars_dict["intake_i"] = treatment_calories

        return odde(t, y, np.array(list(pars_dict.values())), pre_treatment_years=10, treatment_years=2)

    # Run simulation
    sol = solve_ivp(custom_odde, t_span, y0, method="LSODA", t_eval=t_eval, args=(np.array(list(local_pars.values())),))

    # Store weight before BMI conversion
    weights = sol.y[7].copy()

    # Convert weight to BMI
    sol.y[7] = sol.y[7] / (height**2)

    # Print values at key timepoints
    variables = ["Glucose", "Insulin", "FFA", "Si", "Beta", "Sigma", "Inflammation", "BMI", "Weight (kg)"]

    # Initial values (t=0) - only print once
    if not initial_values_printed:
        print("\nInitial values:")
        for var, val in zip(variables[:-1], [y[0] for y in sol.y[:-2]]):  # Exclude height and age
            print(f"{var:12}: {val:.2f}")
        print(f"{'Weight (kg)':12}: {weights[0]:.2f}")
        initial_values_printed = True

    # Post pre-treatment (t=3650, 10 years)
    print("\nPost pre-treatment (10 years):")
    idx_10y = PRE_TREATMENT_DAYS
    for var, vals in zip(variables[:-1], sol.y[:-2]):  # Exclude height and age
        print(f"{var:12}: {vals[idx_10y]:.2f}")
    print(f"{'Weight (kg)':12}: {weights[idx_10y]:.2f}")

    # Final values (t=4380, 12 years)
    print("\nPost treatment (12 years):")
    idx_12y = -1
    for var, vals in zip(variables[:-1], sol.y[:-2]):  # Exclude height and age
        print(f"{var:12}: {vals[idx_12y]:.2f}")
    print(f"{'Weight (kg)':12}: {weights[idx_12y]:.2f}")

    # Calculate DPI at final timepoint
    dpi = 1 - (sol.y[3][idx_12y] * sol.y[4][idx_12y] * sol.y[5][idx_12y]) / (sol.y[3][0] * sol.y[4][0] * sol.y[5][0])
    print(f"{'DPI':12}: {dpi:.2f}")

    return sol


def plot_step_results():
    """Plot biomarkers for STEP trial"""
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

    # Add subplot labels
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

    # STEP trial caloric patterns (example values - adjust based on your calculations)
    pre_calories = 3500  # Estimated maintenance calories for BMI 38.6
    treatment_calories = pre_calories * 0.839  # 16.1% reduction

    # Run simulation and plot results
    sol = simulate_step_trial(pre_calories, treatment_calories)
    years = sol.t / 365

    # Plot each variable
    for i in range(8):
        axs[i].plot(years, sol.y[i], c="#1f77b4", label="STEP 5")
        axs[i].set_xlabel("Time (years)", size="medium")
        axs[i].set_ylabel(vn[i], size="medium", labelpad=2)

    # DPI calculation and plotting
    dpi = 1 - (sol.y[3] * sol.y[4] * sol.y[5]) / (sol.y[3][0] * sol.y[4][0] * sol.y[5][0])
    axs[8].plot(years, dpi, c="#1f77b4", label="STEP 5")
    axs[8].set_xlabel("Time (years)", size="medium")
    axs[8].set_ylabel("DPI", size="medium", labelpad=2)

    # Add legend
    axs[0].legend(fontsize="small", framealpha=0.5, loc="best")

    # Add reference lines to glucose plot
    axs[0].hlines(y=[100, 125], xmin=0, xmax=max(years), linestyles=":", colors=["k", "r"], linewidth=0.8)

    # Match figure size and layout adjustments
    fig.tight_layout()
    fig.subplots_adjust(top=0.88, bottom=0.09, left=0.08, right=0.98, hspace=0.6, wspace=0.45)
    fig.set_size_inches([8.5, 5.7])

    plt.show()


if __name__ == "__main__":
    plot_step_results()
