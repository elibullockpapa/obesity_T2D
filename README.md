# obesity_T2D
The codes for the computational model publsihed in 'Data-driven models show that metabolic profile and progression dynamics of obesity and diabetes determine the success of remission through weight loss'.

model.ode: The ode file that can be run in XPP/AUTO dynamical system analysis ans simulation toolbox (https://sites.pitt.edu/~phase/bard/bardware/xpp/xpp.html)  
model.py: The python code that defines the parameter values and ode function that defines the mathematical formulations given in the manuscript.  
figure2.py: The file uses model.py and generates the Figure 2 given in the mansucript.  

The model has 8 dynamical variables:  
The independent variable is time 't' and is in days.  
g: plasma glucose (mg/dl),  
i: plasma nsulin (uU/ml),  
b: beta cell mass (mg),  
sigma: beta cell function (uU/mg/day)  
Si: whole body insulin sensitivity (uU/ml/day),  
w: body weight (kg),  
inf: systemic inflammation (dimensionless),  
ffa: plasma free fatty acids (umol/l),  

