import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from model_with_wegovy import pars, odde


def find_equilibrium_weight(daily_calories, height_m, initial_weight, simulation_years=5):
    """Simulate weight trajectory until equilibrium for given caloric intake"""

    # Simulation parameters
    t_span = [0, simulation_years * 365]
    t_eval = np.linspace(0, simulation_years * 365, simulation_years * 365 + 1)

    # Initial conditions [g, i, ffa, si, b, sigma, infl, w]
    y0 = [94.1, 9.6, 404, 0.8, 1009, 530, 0.056, initial_weight]

    # Set up parameters
    local_pars = pars.copy()
    local_pars["intake_i"] = daily_calories

    # Run simulation
    sol = solve_ivp(odde, t_span, y0, method="LSODA", t_eval=t_eval, args=(np.array(list(local_pars.values())),))

    # Get final weight and BMI
    final_weight = sol.y[7][-1]
    final_bmi = final_weight / (height_m**2)

    return final_weight, final_bmi, sol.t, sol.y[7]


def print_equilibrium_search(height_m=1.8, initial_weight=81):
    """Print equilibrium weights for different caloric intakes"""

    print(f"\nSearching for equilibrium weights (height: {height_m}m, starting weight: {initial_weight}kg)")
    print("\nDaily Calories | Final Weight | Final BMI")
    print("-" * 45)

    # Test range of daily calories
    for calories in range(2000, 6001, 250):
        final_weight, final_bmi, _, _ = find_equilibrium_weight(calories, height_m, initial_weight)
        print(f"{calories:^13d} | {final_weight:^11.1f} | {final_bmi:^8.1f}")


if __name__ == "__main__":
    print_equilibrium_search()
