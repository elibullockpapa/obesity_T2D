from model_with_wegovy import pars, odde
import numpy as np
from scipy.integrate import solve_ivp


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


def find_equilibrium_calories(target_weight, height_m, pre_treatment_years=5, treatment_years=4, tolerance=0.1):
    """Binary search to find calories needed to maintain target weight"""
    min_calories = 1500
    max_calories = 6000
    initial_weight = 70  # Start from a healthy weight
    years = pre_treatment_years + treatment_years

    while max_calories - min_calories > 1:
        calories = (min_calories + max_calories) / 2

        # Set up simulation parameters
        t_span = [0, years * 365]
        t_eval = np.linspace(0, years * 365, years * 365 + 1)

        # Initial conditions [g, i, ffa, si, b, sigma, infl, w, height, age]
        y0 = [94.1, 9.6, 404, 0.8, 1009, 530, 0.056, initial_weight]

        # Set up parameters
        local_pars = pars.copy()
        local_pars["intake_i"] = calories

        # Run simulation
        def system(t, y):
            return odde(t, y, np.array(list(local_pars.values())), pre_treatment_years, treatment_years)

        sol = solve_ivp(system, t_span, y0, method="LSODA", t_eval=t_eval)

        final_weight = sol.y[7][-1]

        if abs(final_weight - target_weight) < tolerance:
            return calories
        elif final_weight > target_weight:
            max_calories = calories
        else:
            min_calories = calories

    return calories


def analyze_wegovy_targets(trial_name="SELECT", pre_treatment_years=5, treatment_years=4):
    """Analyze caloric requirements for Wegovy trial replication"""
    height_m = pars["height"]
    targets = calculate_target_weights(height_m)
    total_years = pre_treatment_years + treatment_years

    print(
        f"\n{trial_name} Trial Analysis (Pre-treatment: {pre_treatment_years} years, Treatment: {treatment_years} years)"
    )
    print("-" * 100)
    print("BMI Category | Initial Weight | Initial Calories | Final Weight | Final Calories | Weight Change")
    print("-" * 100)

    for category, data in targets.items():
        # Find calories needed for initial weight
        initial_calories = find_equilibrium_calories(
            data["initial_weight"],
            height_m,
            pre_treatment_years=pre_treatment_years,
            treatment_years=treatment_years,
        )

        # Find calories needed for final weight
        final_calories = find_equilibrium_calories(
            data["final_weight"],
            height_m,
            pre_treatment_years=pre_treatment_years,
            treatment_years=treatment_years,
        )

        print(
            f"{category:^11s} | {data['initial_weight']:^14.1f} | {initial_calories:^15.0f} | "
            f"{data['final_weight']:^12.1f} | {final_calories:^14.0f} | {data['reduction']*100:^12.1f}%"
        )

    return targets


def calculate_weight_for_bmi(height_m, bmi):
    """Calculate weight in kg for a given height (m) and BMI"""
    return bmi * (height_m**2)


def calculate_weight_change(initial_weight, target_reduction):
    """Calculate final weight based on a percentage reduction"""
    return initial_weight * (1 - target_reduction / 100)


if __name__ == "__main__":
    # SELECT trial - 9 year simulation (5 pre + 4 treatment)
    targets = analyze_wegovy_targets(trial_name="SELECT", pre_treatment_years=7, treatment_years=4)

    # STEP 5 trial - 12 year simulation (10 pre + 2 treatment)
    height_m = pars["height"]
    target_bmi = 38.6
    target_reduction = 16.1
    simulation_years = 12
    treatment_years = 2
    pre_treatment_years = 10

    initial_weight = calculate_weight_for_bmi(height_m, target_bmi)
    final_weight = calculate_weight_change(initial_weight, target_reduction)

    initial_calories = find_equilibrium_calories(
        initial_weight,
        height_m,
        pre_treatment_years=pre_treatment_years,
        treatment_years=treatment_years,
    )
    final_calories = find_equilibrium_calories(
        final_weight,
        height_m,
        pre_treatment_years=pre_treatment_years,
        treatment_years=treatment_years,
    )

    print(f"\nSTEP 5 Trial Analysis (Pre-treatment: {pre_treatment_years} years, Treatment: {treatment_years} years)")
    print("-" * 100)
    print("Initial BMI | Initial Weight | Initial Calories | Final Weight | Final Calories | Weight Change")
    print("-" * 100)
    print(
        f"{target_bmi:^11.1f} | {initial_weight:^14.1f} | {initial_calories:^15.0f} | "
        f"{final_weight:^12.1f} | {final_calories:^14.0f} | {-target_reduction:^12.1f}%"
    )
