# Modeling Obesity, Diabetes, and GLP-1 Receptor Agonists

**Summary**: This study evaluates and extends a computational model to simulate how GLP-1 receptor agonists (like Wegovy) affect obesity and Type 2 Diabetes, using data from recent clinical trials. While the model successfully captured long-term weight loss patterns, it revealed important limitations in representing the full therapeutic benefits of these medications beyond simple caloric restriction.

## Abstract

The rising prevalence of obesity and Type 2 Diabetes (T2D) represents a major public health challenge, with adult obesity rates in the United States reaching 41.9% by 2020. This study evaluates a computational model, originally developed by Yildirim et al., to simulate the complex pathways linking obesity to T2D development, with particular focus on extending and validating the model against recent clinical trials of GLP-1 receptor agonists. Using data from the SELECT and STEP trials, we determined the caloric restrictions necessary to replicate observed weight loss patterns across different BMI categories (<30, 30-35, 35-40, and ≥40 kg/m²).

Our model successfully captured long-term weight loss trajectories and inflammation reduction patterns observed in clinical trials. However, significant discrepancies emerged in acute metabolic responses, particularly in glucose and insulin dynamics. The model consistently overestimated these parameters, suggesting limitations in its representation of early treatment effects. Most notably, for severely obese patients (BMI ≥40), the model predicted persistent metabolic dysfunction that contradicts clinical evidence of successful treatment outcomes with GLP-1 agonists.

These findings highlight a crucial limitation: only relying on caloric restriction as the sole mechanism of weight loss fails to capture the multiple pathways through which GLP-1 agonists improve metabolic health. Future refinements should incorporate direct effects on insulin sensitivity, beta cell preservation, and inflammation reduction beyond weight loss effects. Despite these limitations, this work provides valuable insights into the interconnected pathways of obesity and T2D, while establishing a framework for evaluating emerging therapeutic strategies beyond simple caloric modification.

## Key Visualizations

### Obesity and T2D Pathway Diagram
![Obesity and T2D Pathway Diagram](https://raw.githubusercontent.com/elibullockpapa/obesity_T2D/refs/heads/main/Report/images/obesity_and_t2d_diagram.png)

### STASIS Trial Simulations
![STASIS Trial Simulations](https://raw.githubusercontent.com/elibullockpapa/obesity_T2D/refs/heads/main/Report/images/stasis_trial_simulations.png)

### STEP Trial Simulations
![STEP Trial Simulations](https://raw.githubusercontent.com/elibullockpapa/obesity_T2D/refs/heads/main/Report/images/step_trial_simulations.png)

### Wegovy Weight Trajectories
![Wegovy Weight Trajectories](https://raw.githubusercontent.com/elibullockpapa/obesity_T2D/refs/heads/main/Report/images/wegovy_weights_plot.png)

## Key Findings and Conclusions

The model demonstrated strong capabilities in simulating long-term weight loss trajectories and inflammation reduction across different BMI categories. However, it revealed important limitations, particularly in representing acute metabolic responses and the full therapeutic benefits of GLP-1 agonists. Future model iterations should incorporate additional mechanisms beyond caloric restriction to better capture the complex metabolic improvements observed in clinical trials.

## References

1. Centers for Disease Control and Prevention. (2024). Adult Obesity Facts.
2. Garvey, W. T., et al. (2022). Two-year effects of semaglutide in adults with overweight or obesity: the STEP 5 trial.
3. Ryan, D.H., et al. (2024). Long-term weight loss effects of semaglutide in obesity without diabetes in the SELECT trial.
4. Yildirim, V., et al. (2023). A data-driven computational model for obesity-driven diabetes onset and remission through weight loss.


