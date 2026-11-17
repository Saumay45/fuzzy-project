"""Exact Table 5 antecedents and singleton consequents."""
import numpy as np

# Each antecedent is either (numeric attribute, linguistic term) or a crisp test.
RULES = [
    ('R1', [('Age', 'High'), ('Oldpeak', 'High')], .95),
    ('R2', [('ExerciseAngina', 'Y'), ('MaxHR', 'Low')], .90),
    ('R3', [('ChestPainType', 'ASY'), ('Oldpeak', 'Medium')], .80),
    ('R4', [('ST_Slope', 'Flat'), ('Oldpeak', 'High')], .95),
    ('R5', [('ST_Slope', 'Down')], .85),
    ('R6', [('Age', 'High'), ('Cholesterol', 'High'), ('RestingBP', 'High')], .75),
    ('R7', [('FastingBS', 1), ('Age', 'High')], .70),
    ('R8', [('Age', 'Low'), ('MaxHR', 'High'), ('Oldpeak', 'Low')], .05),
    ('R9', [('ST_Slope', 'Up'), ('Oldpeak', 'Low'), ('ExerciseAngina', 'N')], .05),
    ('R10', [('ChestPainType', 'ATA'), ('MaxHR', 'High')], .10),
]
CONSEQUENTS = np.array([r[2] for r in RULES])


def rule_description(rule: tuple) -> str:
    """Readable antecedents for UI and educational notes."""
    return ' AND '.join(f'{key} = {value}' for key, value in rule[1])
