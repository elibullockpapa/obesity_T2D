# model_with_wegovy.py

import numpy as np

pars={} 
pars['eg0'] = 24.48  # Endogenous glucose production rate (mg/dl)
pars['k'] = 700  # Glucose elimination constant or sensitivity (dimensionless)
pars['bv'] = 5  # Blood volume (liters) assumed for the insulin and glucose distribution
pars['mmax'] = 1  # Maximum metabolic rate (mg/dl/day)
pars['alpha_m'] = 140  # Half-saturation constant for glucose effect on metabolic rate (mg/dl)
pars['km'] = 2  # Hill coefficient for glucose effect on metabolic rate (dimensionless)
pars['alpha_isr'] = 1.2  # Insulin secretion rate modulation factor (dimensionless)
pars['kisr'] = 2  # Insulin secretion rate sensitivity to metabolic rate changes (dimensionless)
pars['pmax'] = 4.55  # Maximum rate of beta-cell proliferation (mg/day)
pars['kp'] = 4  # Sensitivity parameter for insulin's effect on beta-cell proliferation (dimensionless)
pars['alpha_p'] = 35  # Half-saturation constant for insulin effect on beta-cell proliferation (uU/ml)
pars['p_b'] = 0  # Baseline beta-cell proliferation rate when insulin is minimal (mg/day)
pars['amax'] = 5  # Maximum rate of beta-cell apoptosis (mg/day)
pars['alpha_a'] = 0.37  # Half-saturation constant for glucose effect on beta-cell apoptosis (mg/dl)
pars['ka'] = 6  # Hill coefficient for glucose effect on beta-cell apoptosis (dimensionless)
pars['a_b'] = 0.9  # Baseline apoptosis rate of beta-cells when glucose is minimal (mg/day)
pars['tau_b'] = 1800  # Time constant for beta-cell mass dynamics (days)
pars['height'] = 1.8  # Subject's height (meters)
pars['age_b'] = 30  # Baseline age of the subject (years)
pars['sex'] = 1  # Sex of the subject (1 for male, 0 for female)
pars['cage'] = 0  # Age change coefficient, for dynamic aging effects (dimensionless)
pars['target_si'] = 1.4  # Target whole body insulin sensitivity (uU/ml/day)
pars['tau_si'] = 1  # Time constant for insulin sensitivity dynamics (days)
pars['bmi_h'] = 25  # Threshold BMI for health risk assessment (kg/m^2)
pars['mffa'] = 0.8  # Modulation factor for FFA's effect on insulin sensitivity (dimensionless)
pars['ksi_infl'] = 1.8  # Inflammatory status modulation constant for insulin sensitivity (dimensionless)
pars['ksi_ffa'] = 400  # FFA level modulation constant for insulin sensitivity (umol/l)
pars['nsi_ffa'] = 6  # Hill coefficient for FFA effect on insulin sensitivity (dimensionless)
pars['sw11'] = 1  # Switch parameter to select between different insulin sensitivity calculations (dimensionless)
pars['inc_i1'] = 0  # Increment factor for dietary intake change phase 1 (dimensionless)
pars['inc_i2'] = 0  # Increment factor for dietary intake change phase 2 (dimensionless)
pars['inc_i3'] = 0  # Increment factor for dietary intake change phase 3 (dimensionless)
pars['it1'] = 0  # Start time for dietary intake change phase 1 (days)
pars['it2'] = 100000  # Start time for dietary intake change phase 2 (days)
pars['it3'] = 100000  # Start time for dietary intake change phase 3 (days)
pars['tau_w'] = 1.5  # Time constant for body weight dynamics
pars['k_infl'] = 40  # Inflammation response threshold (dimensionless)
pars['n_infl'] = 6  # Hill coefficient for the effect of BMI on systemic inflammation (dimensionless)
pars['infl_b'] = 0  # Baseline level of systemic inflammation (dimensionless)
pars['tau_infl'] = 1  # Time constant for inflammation dynamics (days)
pars['hgp_bas'] = 2000  # Basal hepatic glucose production (mg/day)
pars['hepa_max'] = 3000  # Maximum capacity of hepatic glucose production beyond basal (mg/day)
pars['hepa_sic'] = 1  # Hepatic insulin sensitivity constant (dimensionless)
pars['a_hgp'] = 4  # Activity coefficient for hepatic glucose production modulation (dimensionless)
pars['gcg'] = 0.1  # Glucagon level (ng/ml)
pars['k_gcg'] = 0  # Modulation coefficient for glucagon effect on hepatic glucose production (dimensionless)
pars['s'] = 0.0002  # Interaction/competition rate between beta-cells (mg^2/day)
pars['png'] = 350  # Rate of beta-cell proliferation by differentiation from other islet cells (mg/day)
pars['sigma_b'] = 536  # Baseline beta-cell function (uU/mg/day)
pars['sgmu'] = 1.5  # Modulation factor for glucose up-regulation of beta-cell function (dimensionless)
pars['sgmd'] = 1  # Modulation factor for glucose down-regulation of beta-cell function (dimensionless)
pars['sfm'] = 1.2  # Modulation factor for FFA effect on beta-cell function (dimensionless)
pars['sim'] = 0.25  # Modulation factor for inflammation effect on beta-cell function (dimensionless)
pars['sgku'] = 81  # Glucose level for up-regulation threshold in beta-cell function modulation (mg/dl)
pars['sgkd'] = 137  # Glucose level for down-regulation threshold in beta-cell function modulation (mg/dl)
pars['sfk'] = 357  # FFA level for modulation threshold in beta-cell function (umol/l)
pars['sik'] = 0.6  # Inflammation level for modulation threshold in beta-cell function (dimensionless)
pars['nsgku'] = 6  # Hill coefficient for glucose effect on up-regulation of beta-cell function (dimensionless)
pars['nsgkd'] = 6  # Hill coefficient for glucose effect on down-regulation of beta-cell function (dimensionless)
pars['nsfk'] = 6  # Hill coefficient for FFA effect on beta-cell function (dimensionless)
pars['nsik'] = 4  # Hill coefficient for inflammation effect on beta-cell function (dimensionless)
pars['tau_sigma'] = 1  # Time constant for dynamics of beta-cell function (days)
pars['sci'] = 1  # Scaling factor for insulin effect in glucose elimination (dimensionless)
pars['gclamp'] = 0  # Glucose clamp level (mg/dl), used in experimental settings
pars['iclamp'] = 0  # Insulin clamp level (uU/ml), used in experimental settings
pars['maxinfl'] = 1  # Maximum level of systemic inflammation allowed in the model (dimensionless)
pars['l0'] = 170  # Baseline lipolysis rate constant (umol/day)
pars['l2'] = 8.1  # Lipolysis response rate to fat mass (umol/day/kg)
pars['cf'] = 2  # Clearance factor for free fatty acids (l/day/kg)
pars['ksif'] = 11  # Sensitivity constant for insulin effect on FFA dynamics (uU/ml)
pars['aa'] = 2  # Activity coefficient for insulin effect on FFA dynamics (dimensionless)
pars['sif'] = 1  # Scaling factor for insulin effect on FFA dynamics (dimensionless)
pars['ffa_ramax'] = 1.106  # Maximum ratio of FFA release to FFA uptake (dimensionless)
pars['lp1'] = 100  # Lower physiological limit for some parameter (unit)
pars['lp2'] = 125  # Upper physiological limit for some parameter (unit)
pars['tau_wsc'] = 1  # Scaling constant for weight change dynamics under different conditions (dimensionless)

pars_l = list(pars.values())  # Convert dictionary values to a list
pars_npa = np.array(pars_l)  # Convert list to numpy array for computational efficiency
pars_n = list(pars.keys())  # Extract dictionary keys for reference

def adjust_for_wegovy(t, pars):
    if t > 1825:  # Assuming drug introduction after three years, expressed in days
        ramp_duration = 365  # duration over which the drug effect ramps up in days
        ramp_factor = min((t - 1825) / ramp_duration, 1)  # caps at 1 when the full effect is reached
        pars['inc_i1'] = max(pars['inc_i1'] - ramp_factor,  - 0.01)   # Gradually apply the reduction in caloric intake
        # pars['tau_w'] += ramp_factor # 

def odde(t,y,pars_npa): 
    heav=lambda x: np.heaviside(x,1) 
    pars = dict(zip(pars_n, pars_npa))

    # Adjust parameters based on the drug effects if applicable
    adjust_for_wegovy(t, pars)

    # Update pars_npa with potentially modified values from pars
    pars_npa = np.array(list(pars.values()))

 #Initial Values 
    g=y[0]
    i=y[1]
    ffa=y[2]
    si=y[3]
    b=y[4]
    sigma=y[5]
    infl=y[6]
    w=y[7]

 #Parameter Values 
    eg0=pars_npa[0]
    k=pars_npa[1]
    bv=pars_npa[2]
    mmax=pars_npa[3]
    alpha_m=pars_npa[4]
    km=pars_npa[5]
    alpha_isr=pars_npa[6]
    kisr=pars_npa[7]
    pmax=pars_npa[8]
    kp=pars_npa[9]
    alpha_p=pars_npa[10]
    p_b=pars_npa[11]
    amax=pars_npa[12]
    alpha_a=pars_npa[13]
    ka=pars_npa[14]
    a_b=pars_npa[15]
    tau_b=pars_npa[16]
    height=pars_npa[17]
    age_b=pars_npa[18]
    sex=pars_npa[19]
    cage=pars_npa[20]
    target_si=pars_npa[21]
    tau_si=pars_npa[22]
    bmi_h=pars_npa[23]
    mffa=pars_npa[24]
    ksi_infl=pars_npa[25]
    ksi_ffa=pars_npa[26]
    nsi_ffa=pars_npa[27]
    sw11=pars_npa[28]
    inc_i1=pars_npa[29]
    inc_i2=pars_npa[30]
    inc_i3=pars_npa[31]
    it1=pars_npa[32]
    it2=pars_npa[33]
    it3=pars_npa[34]
    tau_w=pars_npa[35]
    k_infl=pars_npa[36]
    n_infl=pars_npa[37]
    infl_b=pars_npa[38]
    tau_infl=pars_npa[39]
    hgp_bas=pars_npa[40]
    hepa_max=pars_npa[41]
    hepa_sic=pars_npa[42]
    a_hgp=pars_npa[43]
    gcg=pars_npa[44]
    k_gcg=pars_npa[45]
    s=pars_npa[46]
    png=pars_npa[47]
    sigma_b=pars_npa[48]
    sgmu=pars_npa[49]
    sgmd=pars_npa[50]
    sfm=pars_npa[51]
    sim=pars_npa[52]
    sgku=pars_npa[53]
    sgkd=pars_npa[54]
    sfk=pars_npa[55]
    sik=pars_npa[56]
    nsgku=pars_npa[57]
    nsgkd=pars_npa[58]
    nsfk=pars_npa[59]
    nsik=pars_npa[60]
    tau_sigma=pars_npa[61]
    sci=pars_npa[62]
    gclamp=pars_npa[63]
    iclamp=pars_npa[64]
    maxinfl=pars_npa[65]
    l0=pars_npa[66]
    l2=pars_npa[67]
    cf=pars_npa[68]
    ksif=pars_npa[69]
    aa=pars_npa[70]
    sif=pars_npa[71]
    ffa_ramax=pars_npa[72]
    lp1=pars_npa[73]
    lp2=pars_npa[74]
    tau_wsc=pars_npa[75]
    
#Numerics 
    nsi_infl=6
    bmi = w/height**2
    age = age_b+cage*t/365
    tsi1 = target_si*(ksi_infl/(ksi_infl+infl))*(1-mffa*ffa**nsi_ffa/(ffa**nsi_ffa+ksi_ffa**nsi_ffa))
    tsi2 = target_si*(ksi_infl**nsi_infl/(ksi_infl**nsi_infl+infl**nsi_infl))*(1-mffa*ffa**nsi_ffa/(ffa**nsi_ffa+ksi_ffa**nsi_ffa))
    tsi = sw11*tsi1+(1-sw11)*tsi2
    intake_i = 2500
    inc_int =  0 + heav(t-it1)*heav(it2-t)*heav(it3-t)*inc_i1 + heav(t-it2)*heav(it3-t)*inc_i2 + heav(t-it3)*inc_i3
    
    if t>=it2:
        tau_w=tau_w*tau_wsc
    
    w_base = 81
    expd = 2500/(7700*w_base)
    k_w = 1/7700
    m = mmax*g**km/(alpha_m**km + g**km)
    isr =  sigma*(m)**kisr/(alpha_isr**kisr + (m)**kisr)
    p_isr = p_b + pmax*isr**kp/(alpha_p**kp + isr**kp)
    a_m = a_b + amax*m**ka/(alpha_a**ka + m**ka)
    p =  p_isr
    a =  a_m
    fmass_p = 1.2*bmi + 0.23*age - 16.2
    fmass = w*fmass_p/100
    inc = 1+inc_int
    hepa_si = hepa_sic*si
    hgp = (hgp_bas + hepa_max*(a_hgp+k_gcg*gcg)/((a_hgp+k_gcg*gcg)+hepa_si*i))
    intake = intake_i*inc
    s_glucu = sgmu*g**nsgku/(g**nsgku+sgku**nsgku)
    s_glucd = sgmd*g**nsgkd/(g**nsgkd+sgkd**nsgkd)
    s_ffa = sfm*ffa**nsfk/(ffa**nsfk+sfk**nsfk)
    s_infl = sim*infl**nsik/(infl**nsik+sik**nsik)
    s_inf = sigma_b*(s_glucu - s_glucd*s_ffa - s_infl)
    siff = si*sif
    cl0 = l0*24*60/bv
    cl2 = l2*24*60/bv
    al0 = 2.45*24*60

#Diferetial Equations 
    dg=gclamp+hgp-(eg0+sci*si*i)*g
    di=iclamp+b*isr/bv-k*i
    dffa=(cl0+cl2*fmass)*(ffa_ramax*(ksif**aa)/(ksif**aa+(siff*i)**aa))-cf*w*ffa
    dsi=(tsi-si)/tau_si
    db=(png+p*b-a*b-s*b**2)/tau_b
    dsigma=(s_inf-sigma)/tau_sigma
    dinfl=infl_b+(maxinfl*bmi**n_infl/(bmi**n_infl+k_infl**n_infl)-infl)/tau_infl
    dw=(k_w*intake-expd*w)/tau_w

    dy=np.array([dg,di,dffa,dsi,db,dsigma,dinfl,dw])

    return dy