import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from model_with_wegovy import pars, odde


def simulate_weight_trajectory(initial_weight, pre_treatment_calories, treatment_calories, years=9):
    """Simulate weight trajectory for a given caloric intake pattern"""
    t_span = [0, years * 365]
    t_eval = np.linspace(0, years * 365, years * 365 + 1)

    # Initial conditions [g, i, ffa, si, b, sigma, infl, w]
    y0 = [94.1, 9.6, 404, 0.8, 1009, 530, 0.056, initial_weight]

    # Set up parameters
    local_pars = pars.copy()

    def custom_odde(t, y, pars_npa):
        # Modify intake based on time
        pars_dict = dict(zip(pars.keys(), pars_npa))

        if t < 5 * 365:  # First 5 years - weight gain phase
            pars_dict["intake_i"] = pre_treatment_calories
        else:  # Next 4 years - treatment phase
            pars_dict["intake_i"] = treatment_calories

        return odde(t, y, np.array(list(pars_dict.values())))

    # Run simulation
    sol = solve_ivp(custom_odde, t_span, y0, method="LSODA", t_eval=t_eval, args=(np.array(list(local_pars.values())),))

    return sol.t / 365, sol.y[7]  # Return time in years and weight


def plot_weight_trajectories():
    height_m = pars["height"]
    initial_weight = 81  # Starting from healthy weight (BMI ≈ 25)

    # Target BMIs and their corresponding caloric intakes from outline.txt
    scenarios = {
        "BMI <30": {"target_bmi": 28.0, "pre_calories": 2797, "treatment_calories": 2797 * 0.8948},
        "BMI 30-35": {"target_bmi": 32.5, "pre_calories": 3266, "treatment_calories": 3266 * 0.8821},
        "BMI 35-40": {"target_bmi": 37.5, "pre_calories": 3754, "treatment_calories": 3754 * 0.8799},
        "BMI ≥40": {"target_bmi": 42.0, "pre_calories": 4203, "treatment_calories": 4203 * 0.8777},
    }

    plt.figure(figsize=(12, 8))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for (label, scenario), color in zip(scenarios.items(), colors):
        t, w = simulate_weight_trajectory(
            initial_weight=initial_weight,
            pre_treatment_calories=scenario["pre_calories"],
            treatment_calories=scenario["treatment_calories"],
        )

        plt.plot(t, w, label=label, color=color)

        # Add markers at treatment start (year 5)
        plt.plot(5, w[5 * 365], "o", color=color)
        # Add markers at treatment end (year 9)
        plt.plot(9, w[-1], "s", color=color)

    plt.axvline(x=5, color="gray", linestyle="--", alpha=0.5, label="Treatment Start")
    plt.axvline(x=9, color="gray", linestyle="--", alpha=0.5, label="Treatment End")

    plt.xlabel("Years")
    plt.ylabel("Weight (kg)")
    plt.title("Weight Trajectories by Initial BMI Category\nWith Wegovy Treatment Starting at Year 5")
    plt.grid(True, alpha=0.3)
    plt.legend(title="BMI Category")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_weight_trajectories()
