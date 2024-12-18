import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from model_with_wegovy import pars, odde


def calculate_maintenance_calories(weight_kg, height_m, age=30, sex=1, activity_factor=1.2):
    """Calculate maintenance calories using Mifflin-St Jeor equation"""
    height_cm = height_m * 100
    if sex == 1:  # male
        bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) + 5
    else:  # female
        bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) - 161
    return bmr * activity_factor


def run_wegovy_trial():
    """Simulate weight trajectory for BMI 30-35 group with Wegovy treatment"""
    height_m = pars["height"]

    # We'll only simulate BMI 30-35 group
    target_bmi = 32.5  # middle of 30-35 range
    target_weight = target_bmi * height_m**2

    # Starting from healthy BMI (BMI 25)
    initial_weight = 25 * height_m**2

    print("\nInitial Setup:")
    print(f"Height: {height_m:.2f} m")
    print(f"Initial weight: {initial_weight:.1f} kg (BMI {25:.1f})")
    print(f"Target weight: {target_weight:.1f} kg (BMI {target_bmi:.1f})")

    # Simulation parameters
    t_span = [0, 4380]  # 12 years total
    t_eval = np.linspace(0, 4380, 4381)  # daily points

    # Calculate maintenance calories at target weight
    maintenance_at_target = calculate_maintenance_calories(weight_kg=target_weight, height_m=height_m)

    # To gain weight, we need a surplus
    weight_gain_calories = maintenance_at_target * 2

    print(f"\nCaloric calculations:")
    print(f"Maintenance calories at target: {maintenance_at_target:.0f}")
    print(f"Weight gain calories: {weight_gain_calories:.0f}")

    # Initial conditions [g, i, ffa, si, b, sigma, infl, w]
    y0 = [94.1, 9.6, 404, 0.8, 1009, 530, 0.056, initial_weight]

    local_pars = pars.copy()
    local_pars["intake_i"] = weight_gain_calories

    # Run simulation
    sol = solve_ivp(odde, t_span, y0, method="LSODA", t_eval=t_eval, args=(np.array(list(local_pars.values())),))

    # Print yearly status
    print("\nYearly Status:")
    for year in range(13):  # 0 to 12 years
        day = year * 365
        if day < len(sol.t):
            weight = sol.y[7][day]
            bmi = weight / height_m**2
            maintenance = calculate_maintenance_calories(weight, height_m)

            print(f"\nYear {year}:")
            print(f"Weight: {weight:.1f} kg")
            print(f"BMI: {bmi:.1f}")
            print(f"Maintenance calories: {maintenance:.0f}")

            # Show actual intake (including any reductions)
            actual_intake = weight_gain_calories * (1 + pars["inc_i1"])
            print(f"Actual intake: {actual_intake:.0f}")

            if year >= 8:
                peak_weight = np.max(sol.y[7][:day])
                pct_change = (weight - peak_weight) / peak_weight * 100
                print(f"Change from peak: {pct_change:.1f}%")

    # Calculate final results
    peak_weight = np.max(sol.y[7])
    final_weight = sol.y[7][-1]
    pct_change = (final_weight - peak_weight) / peak_weight * 100

    print("\nFinal Results:")
    print(f"Peak weight: {peak_weight:.1f} kg (BMI {peak_weight/height_m**2:.1f})")
    print(f"Final weight: {final_weight:.1f} kg (BMI {final_weight/height_m**2:.1f})")
    print(f"Total change from peak: {pct_change:.1f}%")

    # Plot results
    plt.figure(figsize=(12, 8))

    # Convert to years for plotting
    years = sol.t / 365

    # Plot absolute weight instead of percentage change
    plt.plot(years, sol.y[7], label="BMI 30-35")
    plt.axvline(x=8, color="gray", linestyle="--", alpha=0.5, label="Wegovy Start")
    plt.grid(True, alpha=0.3)
    plt.xlabel("Time (years)")
    plt.ylabel("Weight (kg)")
    plt.title("Simulated Wegovy Trial Results (BMI 30-35)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    run_wegovy_trial()
