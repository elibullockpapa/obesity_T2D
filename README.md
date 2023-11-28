# obesity_T2D
The codes for the computational model that was published in the paper titled 'A data-driven computational model for obesity-driven diabetes onset and remission through weight loss' (https://doi.org/10.1016/j.isci.2023.108324).

Yildirim, V., Sheraton, V.M., Brands, R., Crielaard, L., Quax, R., van Riel, N.A.W., Stronks, K., Nicolaou, M., and Sloot, P.M.A. (2023). A data-driven computational model for obesity-driven diabetes onset and remission through weight loss. iScience 26, 108324. 10.1016/j.isci.2023.108324.


model.ode: The ode file that can be run in XPP/AUTO dynamical system analysis ans simulation toolbox (https://sites.pitt.edu/~phase/bard/bardware/xpp/xpp.html)  
model.py: The python code that defines the parameter values and ode function that defines the mathematical formulations given in the manuscript.  
figure2.py: The file uses model.py and generates the Figure 2 given in the mansucript.  

The model has 8 variables:  
The independent variable is time 't' and is in days.  
g: plasma glucose (mg/dl),  
i: plasma nsulin (uU/ml),  
b: beta cell mass (mg),  
sigma: beta cell function (uU/mg/day)  
Si: whole body insulin sensitivity (uU/ml/day),  
w: body weight (kg),  
inf: systemic inflammation (dimensionless),  
ffa: plasma free fatty acids (umol/l),  

