import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from model_with_wegovy import pars, odde
import string
import matplotlib.transforms as mtransforms
from find_caloric_intake_in_trial import find_equilibrium_calories

# Simulation time periods (in days)
PRE_TREATMENT_DAYS = 0  # no pre-treatment
TREATMENT_DAYS = 730  # 2 years
TOTAL_DAYS = PRE_TREATMENT_DAYS + TREATMENT_DAYS
CALORIE_REDUCTION_RAMP_DAYS = 60  # 2 months

# Initial values measured in STEP trial
INITIAL_GLUCOSE = 95.4  # mg/dl (default is 94.1)
INITIAL_INSULIN = 12.6  # μU/ml (default is 9.6)
INITIAL_WEIGHT = 105.6  # kg (default is 81)
INITIAL_HEIGHT = 1.65  # m (default is 1.8)
INITIAL_AGE = 47.3  # years (default is 30)

# Values that fit the model after 2 weeks (default values in parentheses)
# These values were not measured in the trial, so we fit them to the model
INITIAL_FFA = 688.65  # μmol/l (was 400)
INITIAL_SI = 0.26  # ml/μU/day (was 0.8)
INITIAL_BETA = 1009  # mg (unchanged)
INITIAL_SIGMA = 501.64  # μU/mg/day (was 530)
INITIAL_INFLAMMATION = 0.45  # dimensionless (was 0.056)

# Post treatment values measured in STEP trial
FINAL_WEIGHT = INITIAL_WEIGHT * (1 - 0.152)  # kg (-15.2%)
FINAL_GLUCOSE = INITIAL_GLUCOSE - 7.2  # mg/dl (-0.4 mmol/L)
FINAL_INSULIN = INITIAL_INSULIN * (1 - 0.327)  # μU/ml (-32.7%)
FINAL_FFA = INITIAL_FFA * 1.003  # μmol/l (+0.3%)
FINAL_INFLAMMATION = INITIAL_INFLAMMATION * (1 - 0.567)  # dimensionless (-56.7%)

initial_values_printed = False


def simulate_step_trial(initial_weight=INITIAL_WEIGHT, height=INITIAL_HEIGHT, age=INITIAL_AGE):
    """Simulate STEP trial with pre-treatment and treatment phases"""
    global initial_values_printed

    t_span = [0, TREATMENT_DAYS]
    t_eval = np.linspace(0, TREATMENT_DAYS, TREATMENT_DAYS + 1)

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

    # Calculate equilibrium calories for initial and final weights
    initial_calories, _ = find_equilibrium_calories(initial_weight, y0, simulation_years=5)
    final_calories, _ = find_equilibrium_calories(FINAL_WEIGHT, y0, simulation_years=2)
    print(
        f"Original calorie consumption: {initial_calories:.0f}, Post treatment calorie consumption: {final_calories:.0f}"
    )

    # Set up parameters
    local_pars = pars.copy()

    def custom_odde(t, y, pars_npa):
        pars_dict = dict(zip(pars.keys(), pars_npa))

        if t <= CALORIE_REDUCTION_RAMP_DAYS:  # 2-month ramp up
            ramp_factor = t / CALORIE_REDUCTION_RAMP_DAYS  # Linear ramp from 0 to 1
            pars_dict["intake_i"] = initial_calories - (initial_calories - final_calories) * ramp_factor
        else:  # Full treatment
            pars_dict["intake_i"] = final_calories

        return odde(t, y, np.array(list(pars_dict.values())), pre_treatment_years=0, treatment_years=2)

    # Run simulation
    sol = solve_ivp(custom_odde, t_span, y0, method="LSODA", t_eval=t_eval, args=(np.array(list(local_pars.values())),))

    # Store weight before BMI conversion
    weights = sol.y[7].copy()

    # Convert weight to BMI
    sol.y[7] = sol.y[7] / (height**2)

    # Print values at key timepoints
    variables = ["Glucose", "Insulin", "FFA", "Si", "Beta", "Sigma", "Inflammation", "BMI", "Weight (kg)"]

    # Initial values (t=0)
    print("\nInitial values:")
    for var, val in zip(variables[:-1], [y[0] for y in sol.y[:-2]]):  # Exclude height and age
        print(f"{var:12}: {val:.2f}")
    print(f"{'Weight (kg)':12}: {weights[0]:.2f}")

    # After 2 weeks
    print("\nValues after 2 weeks:")
    idx_2w = 14  # 14 days
    for var, vals in zip(variables[:-1], sol.y[:-2]):  # Exclude height and age
        print(f"{var:12}: {vals[idx_2w]:.2f}")
    print(f"{'Weight (kg)':12}: {weights[idx_2w]:.2f}")

    # Post ramp-up (2 months)
    print("\nPost ramp-up (2 months):")
    idx_2m = CALORIE_REDUCTION_RAMP_DAYS
    for var, vals in zip(variables[:-1], sol.y[:-2]):  # Exclude height and age
        print(f"{var:12}: {vals[idx_2m]:.2f}")
    print(f"{'Weight (kg)':12}: {weights[idx_2m]:.2f}")

    # Final values (2 years)
    print("\nFinal values (2 years):")
    idx_final = -1
    for var, vals in zip(variables[:-1], sol.y[:-2]):  # Exclude height and age
        print(f"{var:12}: {vals[idx_final]:.2f}")
    print(f"{'Weight (kg)':12}: {weights[idx_final]:.2f}")

    # Calculate DPI at final timepoint
    dpi = 1 - (sol.y[3][idx_final] * sol.y[4][idx_final] * sol.y[5][idx_final]) / (
        sol.y[3][0] * sol.y[4][0] * sol.y[5][0]
    )
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

    # Run simulation and plot results
    sol = simulate_step_trial()
    years = sol.t / 365

    # Plot each variable
    for i in range(8):
        axs[i].plot(years, sol.y[i], c="#1f77b4", label="Simulation")

        # Add measured initial and final points where available
        if i == 0:  # Glucose
            axs[i].scatter(0, INITIAL_GLUCOSE, c="red", label="Measured")
            axs[i].scatter(2, FINAL_GLUCOSE, c="red")
        elif i == 1:  # Insulin
            axs[i].scatter(0, INITIAL_INSULIN, c="red")
            axs[i].scatter(2, FINAL_INSULIN, c="red")
        elif i == 2:  # FFA
            axs[i].scatter(2, FINAL_FFA, c="red")
        elif i == 6:  # Inflammation
            axs[i].scatter(2, FINAL_INFLAMMATION, c="red")
        elif i == 7:  # BMI
            initial_bmi = INITIAL_WEIGHT / (INITIAL_HEIGHT**2)
            final_bmi = FINAL_WEIGHT / (INITIAL_HEIGHT**2)
            axs[i].scatter(0, initial_bmi, c="red")
            axs[i].scatter(2, final_bmi, c="red")

        axs[i].set_xlabel("Time (years)", size="medium")
        axs[i].set_ylabel(vn[i], size="medium", labelpad=2)

    # DPI calculation and plotting
    dpi = 1 - (sol.y[3] * sol.y[4] * sol.y[5]) / (sol.y[3][0] * sol.y[4][0] * sol.y[5][0])
    axs[8].plot(years, dpi, c="#1f77b4", label="STEP 5")
    axs[8].set_xlabel("Time (years)", size="medium")
    axs[8].set_ylabel("DPI", size="medium", labelpad=2)

    # Add a single legend above all subplots
    fig.legend(
        ["Simulation", "Measured"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=2,
        fontsize="small",
        framealpha=0.5,
    )

    # First apply tight_layout
    fig.tight_layout()

    # Then adjust the top margin for the legend
    # Values between 0 and 1, where 0.9 means 90% of height is for plots
    fig.subplots_adjust(top=0.85)

    fig.set_size_inches([8.5, 5.7])

    plt.show()


if __name__ == "__main__":
    plot_step_results()
