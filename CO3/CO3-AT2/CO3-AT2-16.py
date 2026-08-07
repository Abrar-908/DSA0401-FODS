disease = 0.02
sensitivity = 0.95
specificity = 0.90

no_disease = 1 - disease
false_positive = 1 - specificity

positive = (sensitivity * disease) + (false_positive * no_disease)

posterior = (sensitivity * disease) / positive

print("P(Disease | Positive Test) =", posterior)