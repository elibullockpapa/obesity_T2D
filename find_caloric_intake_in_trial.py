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


def find_equilibrium_calories(target_weight, y0, simulation_years=5, tolerance=0.1):
    """Binary search to find calories needed to maintain target weight and return final state

    Args:
        target_weight: Target weight in kg
        y0: Initial state array [g, i, ffa, si, b, sigma, infl, w, height, age]
        simulation_years: Number of years to simulate to reach equilibrium (default 5)
        tolerance: Acceptable difference between final and target weight (default 0.1)

    Returns:
        tuple: (calories, final_y_array)
    """
    min_calories = 1500
    max_calories = 6000

    while max_calories - min_calories > 1:
        calories = (min_calories + max_calories) / 2

        # Set up simulation parameters
        t_span = [0, simulation_years * 365]
        t_eval = np.linspace(0, simulation_years * 365, simulation_years * 365 + 1)

        # Set up parameters
        local_pars = pars.copy()
        local_pars["intake_i"] = calories

        # Run simulation without Wegovy treatment
        def system(t, y):
            return odde(t, y, np.array(list(local_pars.values())), 0, 0)

        sol = solve_ivp(system, t_span, y0, method="LSODA", t_eval=t_eval)

        final_weight = sol.y[7][-1]

        if abs(final_weight - target_weight) < tolerance:
            return calories, sol.y[:, -1]
        elif final_weight > target_weight:
            max_calories = calories
        else:
            min_calories = calories

    return calories, sol.y[:, -1]


def analyze_select_trial(pre_treatment_years=7, treatment_years=4):
    """Analyze caloric requirements for SELECT trial"""
    height_m = 1.65
    targets = calculate_target_weights(height_m)

    # Initial conditions specific to SELECT trial
    # [g, i, ffa, si, b, sigma, infl, w, height, age]
    base_y0 = [94.1, 9.6, 404, 0.8, 1009, 530, 0.056, 70, height_m, 47.3]

    print(f"\nSELECT Trial Analysis (Pre-treatment: {pre_treatment_years} years, Treatment: {treatment_years} years)")
    print("-" * 100)
    print("BMI Category | Initial Weight | Initial Calories | Final Weight | Final Calories | Weight Change")
    print("-" * 100)

    results = {}
    for category, data in targets.items():
        # Find calories needed for initial weight
        y0 = base_y0.copy()
        initial_calories, final_state = find_equilibrium_calories(
            data["initial_weight"], y0, simulation_years=pre_treatment_years
        )

        # Use final state from initial simulation as starting point for final weight simulation
        final_calories, _ = find_equilibrium_calories(
            data["final_weight"], final_state, simulation_years=treatment_years
        )

        results[category] = {
            "initial_weight": data["initial_weight"],
            "initial_calories": initial_calories,
            "final_weight": data["final_weight"],
            "final_calories": final_calories,
            "weight_change": data["reduction"] * 100,
        }

        print(
            f"{category:^11s} | {data['initial_weight']:^14.1f} | {initial_calories:^15.0f} | "
            f"{data['final_weight']:^12.1f} | {final_calories:^14.0f} | {data['reduction']*100:^12.1f}%"
        )

    return results


def analyze_step_trial(pre_treatment_years=7, treatment_years=2):
    """Analyze caloric requirements for STEP 5 trial"""
    # STEP trial specific parameters
    target_bmi = 38.6
    target_reduction = 16.1
    height_m = 1.65

    # Initial conditions specific to STEP trial
    # [g, i, ffa, si, b, sigma, infl, w, height, age]
    base_y0 = [92.0, 9.8, 400, 0.75, 1000, 525, 0.052, 68, height_m, 45.0]

    initial_weight = calculate_weight_for_bmi(height_m, target_bmi)
    final_weight = calculate_weight_change(initial_weight, target_reduction)

    # Find initial equilibrium
    y0 = base_y0.copy()
    initial_calories, final_state = find_equilibrium_calories(initial_weight, y0, simulation_years=pre_treatment_years)

    # Find final equilibrium starting from the final state of initial simulation
    final_calories, _ = find_equilibrium_calories(final_weight, final_state, simulation_years=treatment_years)

    print(f"\nSTEP 5 Trial Analysis (Pre-treatment: {pre_treatment_years} years, Treatment: {treatment_years} years)")
    print("-" * 100)
    print("Initial BMI | Initial Weight | Initial Calories | Final Weight | Final Calories | Weight Change")
    print("-" * 100)
    print(
        f"{target_bmi:^11.1f} | {initial_weight:^14.1f} | {initial_calories:^15.0f} | "
        f"{final_weight:^12.1f} | {final_calories:^14.0f} | {-target_reduction:^12.1f}%"
    )

    return {
        "initial_bmi": target_bmi,
        "initial_weight": initial_weight,
        "initial_calories": initial_calories,
        "final_weight": final_weight,
        "final_calories": final_calories,
        "weight_change": -target_reduction,
    }


def calculate_weight_for_bmi(height_m, bmi):
    """Calculate weight in kg for a given height (m) and BMI"""
    return bmi * (height_m**2)


def calculate_weight_change(initial_weight, target_reduction):
    """Calculate final weight based on a percentage reduction"""
    return initial_weight * (1 - target_reduction / 100)


if __name__ == "__main__":
    # Run both trial analyses
    select_results = analyze_select_trial()
    step_results = analyze_step_trial()
