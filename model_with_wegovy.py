# model_with_wegovy.py

import numpy as np

pars = {}
pars["eg0"] = 24.48  # Endogenous glucose production rate (mg/dl)
pars["k"] = 700  # Glucose elimination constant or sensitivity (dimensionless)
pars["bv"] = 5  # Blood volume (liters) assumed for the insulin and glucose distribution
pars["mmax"] = 1  # Maximum metabolic rate (mg/dl/day)
pars["alpha_m"] = 140  # Half-saturation constant for glucose effect on metabolic rate (mg/dl)
pars["km"] = 2  # Hill coefficient for glucose effect on metabolic rate (dimensionless)
pars["alpha_isr"] = 1.2  # Insulin secretion rate modulation factor (dimensionless)
pars["kisr"] = 2  # Insulin secretion rate sensitivity to metabolic rate changes (dimensionless)
pars["pmax"] = 4.55  # Maximum rate of beta-cell proliferation (mg/day)
pars["kp"] = 4  # Sensitivity parameter for insulin's effect on beta-cell proliferation (dimensionless)
pars["alpha_p"] = 35  # Half-saturation constant for insulin effect on beta-cell proliferation (uU/ml)
pars["p_b"] = 0  # Baseline beta-cell proliferation rate when insulin is minimal (mg/day)
pars["amax"] = 5  # Maximum rate of beta-cell apoptosis (mg/day)
pars["alpha_a"] = 0.37  # Half-saturation constant for glucose effect on beta-cell apoptosis (mg/dl)
pars["ka"] = 6  # Hill coefficient for glucose effect on beta-cell apoptosis (dimensionless)
pars["a_b"] = 0.9  # Baseline apoptosis rate of beta-cells when glucose is minimal (mg/day)
pars["tau_b"] = 1800  # Time constant for beta-cell mass dynamics (days)
pars["sex"] = 0  # Sex of the subject (1 for male, 0 for female), these studies have more females so we will use 0
pars["cage"] = 0  # Age change coefficient, for dynamic aging effects (dimensionless)
pars["target_si"] = 1.4  # Target whole body insulin sensitivity (uU/ml/day)
pars["tau_si"] = 1  # Time constant for insulin sensitivity dynamics (days)
pars["bmi_h"] = 25  # Threshold BMI for health risk assessment (kg/m^2)
pars["mffa"] = 0.8  # Modulation factor for FFA's effect on insulin sensitivity (dimensionless)
pars["ksi_infl"] = 1.8  # Inflammatory status modulation constant for insulin sensitivity (dimensionless)
pars["ksi_ffa"] = 400  # FFA level modulation constant for insulin sensitivity (umol/l)
pars["nsi_ffa"] = 6  # Hill coefficient for FFA effect on insulin sensitivity (dimensionless)
pars["sw11"] = 1  # Switch parameter to select between different insulin sensitivity calculations (dimensionless)
pars["inc_i1"] = 0  # Increment factor for dietary intake change phase 1 (dimensionless)
pars["inc_i2"] = 0  # Increment factor for dietary intake change phase 2 (dimensionless)
pars["inc_i3"] = 0  # Increment factor for dietary intake change phase 3 (dimensionless)
pars["it1"] = 0  # Start time for dietary intake change phase 1 (days)
pars["it2"] = 100000  # Start time for dietary intake change phase 2 (days)
pars["it3"] = 100000  # Start time for dietary intake change phase 3 (days)
pars["tau_w"] = 1.5  # Time constant for body weight dynamics
pars["k_infl"] = 40  # Inflammation response threshold (dimensionless)
pars["n_infl"] = 6  # Hill coefficient for the effect of BMI on systemic inflammation (dimensionless)
pars["infl_b"] = 0  # Baseline level of systemic inflammation (dimensionless)
pars["tau_infl"] = 1  # Time constant for inflammation dynamics (days)
pars["hgp_bas"] = 2000  # Basal hepatic glucose production (mg/day)
pars["hepa_max"] = 3000  # Maximum capacity of hepatic glucose production beyond basal (mg/day)
pars["hepa_sic"] = 1  # Hepatic insulin sensitivity constant (dimensionless)
pars["a_hgp"] = 4  # Activity coefficient for hepatic glucose production modulation (dimensionless)
pars["gcg"] = 0.1  # Glucagon level (ng/ml)
pars["k_gcg"] = 0  # Modulation coefficient for glucagon effect on hepatic glucose production (dimensionless)
pars["s"] = 0.0002  # Interaction/competition rate between beta-cells (mg^2/day)
pars["png"] = 350  # Rate of beta-cell proliferation by differentiation from other islet cells (mg/day)
pars["sigma_b"] = 536  # Baseline beta-cell function (uU/mg/day)
pars["sgmu"] = 1.5  # Modulation factor for glucose up-regulation of beta-cell function (dimensionless)
pars["sgmd"] = 1  # Modulation factor for glucose down-regulation of beta-cell function (dimensionless)
pars["sfm"] = 1.2  # Modulation factor for FFA effect on beta-cell function (dimensionless)
pars["sim"] = 0.25  # Modulation factor for inflammation effect on beta-cell function (dimensionless)
pars["sgku"] = 81  # Glucose level for up-regulation threshold in beta-cell function modulation (mg/dl)
pars["sgkd"] = 137  # Glucose level for down-regulation threshold in beta-cell function modulation (mg/dl)
pars["sfk"] = 357  # FFA level for modulation threshold in beta-cell function (umol/l)
pars["sik"] = 0.6  # Inflammation level for modulation threshold in beta-cell function (dimensionless)
pars["nsgku"] = 6  # Hill coefficient for glucose effect on up-regulation of beta-cell function (dimensionless)
pars["nsgkd"] = 6  # Hill coefficient for glucose effect on down-regulation of beta-cell function (dimensionless)
pars["nsfk"] = 6  # Hill coefficient for FFA effect on beta-cell function (dimensionless)
pars["nsik"] = 4  # Hill coefficient for inflammation effect on beta-cell function (dimensionless)
pars["tau_sigma"] = 1  # Time constant for dynamics of beta-cell function (days)
pars["sci"] = 1  # Scaling factor for insulin effect in glucose elimination (dimensionless)
pars["gclamp"] = 0  # Glucose clamp level (mg/dl), used in experimental settings
pars["iclamp"] = 0  # Insulin clamp level (uU/ml), used in experimental settings
pars["maxinfl"] = 1  # Maximum level of systemic inflammation allowed in the model (dimensionless)
pars["l0"] = 170  # Baseline lipolysis rate constant (umol/day)
pars["l2"] = 8.1  # Lipolysis response rate to fat mass (umol/day/kg)
pars["cf"] = 2  # Clearance factor for free fatty acids (l/day/kg)
pars["ksif"] = 11  # Sensitivity constant for insulin effect on FFA dynamics (uU/ml)
pars["aa"] = 2  # Activity coefficient for insulin effect on FFA dynamics (dimensionless)
pars["sif"] = 1  # Scaling factor for insulin effect on FFA dynamics (dimensionless)
pars["ffa_ramax"] = 1.106  # Maximum ratio of FFA release to FFA uptake (dimensionless)
pars["lp1"] = 100  # Lower physiological limit for some parameter (unit)
pars["lp2"] = 125  # Upper physiological limit for some parameter (unit)
pars["tau_wsc"] = 1  # Scaling constant for weight change dynamics under different conditions (dimensionless)
pars["intake_i"] = 2500  # Initial daily caloric intake (calories/day)

pars_l = list(pars.values())  # Convert dictionary values to a list
pars_npa = np.array(pars_l)  # Convert list to numpy array for computational efficiency
pars_n = list(pars.keys())  # Extract dictionary keys for reference


def adjust_for_wegovy(t, y, pars, pre_treatment_years=1, treatment_years=4):
    """Adjusts model parameters to simulate Wegovy treatment

    Args:
        t: Current simulation time in days
        y: Current state variables (includes height at y[8] and age at y[9])
        pars: Model parameters
        pre_treatment_years: Number of years before treatment starts (default 1)
        treatment_years: Number of years of treatment (default 4)
    """
    # Reset inc_i1 to 0 by default
    pars["inc_i1"] = 0

    treatment_start = pre_treatment_years * 365
    treatment_end = treatment_start + (treatment_years * 365)

    if t > treatment_start and t < treatment_end:
        ramp_duration = 365  # Ramp up over 1 year
        ramp_factor = min((t - treatment_start) / ramp_duration, 1)

        current_weight = y[7]
        height = y[8]  # Height now comes from state variables
        bmi = current_weight / (height**2)

        # Target reduction based on BMI category (simplified from trial data)
        if bmi < 30:
            target_reduction = -0.1052
        elif bmi < 35:
            target_reduction = -0.1179
        elif bmi < 40:
            target_reduction = -0.1201
        else:
            target_reduction = -0.1223

        # Apply reduction directly to intake
        pars["inc_i1"] = target_reduction * ramp_factor


def odde(t, y, pars_npa, pre_treatment_years=1, treatment_years=4):
    """ODE system for obesity and T2D model with Wegovy treatment

    Args:
        t: Time (days)
        y: State variables [g, i, ffa, si, b, sigma, infl, w, height, age]
        pars_npa: Model parameters as numpy array
        pre_treatment_years: Years before treatment starts
        treatment_years: Years of treatment
    """

    def heav(x):
        return np.heaviside(x, 1)

    # Convert parameters array to dictionary
    pars_dict = dict(zip(pars.keys(), pars_npa))

    # Adjust parameters based on the drug effects if applicable
    adjust_for_wegovy(t, y, pars_dict, pre_treatment_years, treatment_years)

    # Update pars_npa with potentially modified values
    pars_npa = np.array(list(pars_dict.values()))

    # Extract state variables
    g, i, ffa, si, b, sigma, infl, w, height, age = y

    # Update age based on simulation time
    current_age = age + pars_dict["cage"] * t / 365

    # Calculate BMI
    bmi = w / height**2

    # Extract key parameters
    eg0 = pars_dict["eg0"]
    k = pars_dict["k"]
    bv = pars_dict["bv"]
    mmax = pars_dict["mmax"]
    alpha_m = pars_dict["alpha_m"]
    km = pars_dict["km"]
    alpha_isr = pars_dict["alpha_isr"]
    kisr = pars_dict["kisr"]
    pmax = pars_dict["pmax"]
    kp = pars_dict["kp"]
    alpha_p = pars_dict["alpha_p"]
    p_b = pars_dict["p_b"]
    amax = pars_dict["amax"]
    alpha_a = pars_dict["alpha_a"]
    ka = pars_dict["ka"]
    a_b = pars_dict["a_b"]
    tau_b = pars_dict["tau_b"]
    target_si = pars_dict["target_si"]
    tau_si = pars_dict["tau_si"]
    mffa = pars_dict["mffa"]
    ksi_infl = pars_dict["ksi_infl"]
    ksi_ffa = pars_dict["ksi_ffa"]
    nsi_ffa = pars_dict["nsi_ffa"]
    sw11 = pars_dict["sw11"]
    inc_i1 = pars_dict["inc_i1"]
    inc_i2 = pars_dict["inc_i2"]
    inc_i3 = pars_dict["inc_i3"]
    it1 = pars_dict["it1"]
    it2 = pars_dict["it2"]
    it3 = pars_dict["it3"]
    tau_w = pars_dict["tau_w"]
    k_infl = pars_dict["k_infl"]
    n_infl = pars_dict["n_infl"]
    infl_b = pars_dict["infl_b"]
    tau_infl = pars_dict["tau_infl"]
    hgp_bas = pars_dict["hgp_bas"]
    hepa_max = pars_dict["hepa_max"]
    hepa_sic = pars_dict["hepa_sic"]
    a_hgp = pars_dict["a_hgp"]
    gcg = pars_dict["gcg"]
    k_gcg = pars_dict["k_gcg"]
    s = pars_dict["s"]
    png = pars_dict["png"]
    sigma_b = pars_dict["sigma_b"]
    sgmu = pars_dict["sgmu"]
    sgmd = pars_dict["sgmd"]
    sfm = pars_dict["sfm"]
    sim = pars_dict["sim"]
    sgku = pars_dict["sgku"]
    sgkd = pars_dict["sgkd"]
    sfk = pars_dict["sfk"]
    sik = pars_dict["sik"]
    nsgku = pars_dict["nsgku"]
    nsgkd = pars_dict["nsgkd"]
    nsfk = pars_dict["nsfk"]
    nsik = pars_dict["nsik"]
    tau_sigma = pars_dict["tau_sigma"]
    sci = pars_dict["sci"]
    gclamp = pars_dict["gclamp"]
    iclamp = pars_dict["iclamp"]
    maxinfl = pars_dict["maxinfl"]
    l0 = pars_dict["l0"]
    l2 = pars_dict["l2"]
    cf = pars_dict["cf"]
    ksif = pars_dict["ksif"]
    aa = pars_dict["aa"]
    sif = pars_dict["sif"]
    ffa_ramax = pars_dict["ffa_ramax"]
    tau_wsc = pars_dict["tau_wsc"]

    # Numerics
    nsi_infl = 6
    tsi1 = target_si * (ksi_infl / (ksi_infl + infl)) * (1 - mffa * ffa**nsi_ffa / (ffa**nsi_ffa + ksi_ffa**nsi_ffa))
    tsi2 = (
        target_si
        * (ksi_infl**nsi_infl / (ksi_infl**nsi_infl + infl**nsi_infl))
        * (1 - mffa * ffa**nsi_ffa / (ffa**nsi_ffa + ksi_ffa**nsi_ffa))
    )
    tsi = sw11 * tsi1 + (1 - sw11) * tsi2
    intake_i = pars_dict["intake_i"]
    inc_int = (
        0
        + heav(t - it1) * heav(it2 - t) * heav(it3 - t) * inc_i1
        + heav(t - it2) * heav(it3 - t) * inc_i2
        + heav(t - it3) * inc_i3
    )

    if t >= it2:
        tau_w = tau_w * tau_wsc

    w_base = 81
    expd = 2500 / (7700 * w_base)
    k_w = 1 / 7700
    m = mmax * g**km / (alpha_m**km + g**km)
    isr = sigma * (m) ** kisr / (alpha_isr**kisr + (m) ** kisr)
    p_isr = p_b + pmax * isr**kp / (alpha_p**kp + isr**kp)
    a_m = a_b + amax * m**ka / (alpha_a**ka + m**ka)
    p = p_isr
    a = a_m
    fmass_p = 1.2 * bmi + 0.23 * current_age - 16.2
    fmass = w * fmass_p / 100
    inc = 1 + inc_int
    hepa_si = hepa_sic * si
    hgp = hgp_bas + hepa_max * (a_hgp + k_gcg * gcg) / ((a_hgp + k_gcg * gcg) + hepa_si * i)
    intake = intake_i * inc
    s_glucu = sgmu * g**nsgku / (g**nsgku + sgku**nsgku)
    s_glucd = sgmd * g**nsgkd / (g**nsgkd + sgkd**nsgkd)
    s_ffa = sfm * ffa**nsfk / (ffa**nsfk + sfk**nsfk)
    s_infl = sim * infl**nsik / (infl**nsik + sik**nsik)
    s_inf = sigma_b * (s_glucu - s_glucd * s_ffa - s_infl)
    siff = si * sif
    cl0 = l0 * 24 * 60 / bv
    cl2 = l2 * 24 * 60 / bv

    # Differential Equations
    dg = gclamp + hgp - (eg0 + sci * si * i) * g
    di = iclamp + b * isr / bv - k * i
    dffa = (cl0 + cl2 * fmass) * (ffa_ramax * (ksif**aa) / (ksif**aa + (siff * i) ** aa)) - cf * w * ffa
    dsi = (tsi - si) / tau_si
    db = (png + p * b - a * b - s * b**2) / tau_b
    dsigma = (s_inf - sigma) / tau_sigma
    dinfl = infl_b + (maxinfl * bmi**n_infl / (bmi**n_infl + k_infl**n_infl) - infl) / tau_infl
    dw = (k_w * intake - expd * w) / tau_w

    # Return derivatives for all state variables including height and age (which don't change)
    dy = np.array([dg, di, dffa, dsi, db, dsigma, dinfl, dw, 0, 0])

    return dy
