
""" First step import Pyomo library and declare solver """
import matplotlib.pyplot as plt
import pandas as pd
import pyomo.environ as pyo
solver = 'appsi_highs'
SOLVER = pyo.SolverFactory(solver)
assert SOLVER.available(), f"Solver {solver} is not available."

""" Second step """
# create model with optional problem title
model = pyo.ConcreteModel("Production Planning: Version 1")
# display model
model.display()

"""" third step """
# create decision variables
model.x_M = pyo.Var(bounds=(0, None))
model.x_A = pyo.Var(bounds=(0, 80))
model.x_B = pyo.Var(bounds=(0, 100))

model.y_U = pyo.Var(bounds=(0, 40))
model.y_V = pyo.Var(bounds=(0, None))

# display updated model
model.display()

""" fourth step """
# create expressions 
model.revenue = 270 * model.y_U + 210 * model.y_V
model.cost = 10 * model.x_M + 50 * model.x_A + 40 * model.x_B

# expressions can be printed
print(model.revenue)
print(model.cost)

""" fifth step """
# create objective function
model.profit = pyo.Objective(expr=model.revenue - model.cost, sense=pyo.maximize)

""" Sixth step """
# create constraints Logical realations between variables
model.raw_materials = pyo.Constraint(expr = 10 * model.y_U + 9 * model.y_V <= model.x_M)
model.labor_A = pyo.Constraint(expr = 1 * model.y_U + 1 * model.y_V <= model.x_A)
model.labor_B = pyo.Constraint(expr = 2 * model.y_U + 1 * model.y_V <= model.x_B)

""" Seventh step """
# Solve the model using a solver
results = SOLVER.solve(model, tee=True)

"""Eighth step"""
# display results and report the solution

# display the whole model
model.pprint()

# display a component of the model
model.profit.pprint()

pyo.value(model.profit)

print(f" Profit = {pyo.value(model.profit): 9.2f}")
print(f"Revenue = {pyo.value(model.revenue): 9.2f}")
print(f"   Cost = {pyo.value(model.cost): 9.2f}")

print("x_A =", model.x_A())
print("x_B =", model.x_B())
print("x_M =", model.x_M())

""" Aditional step report with pandas """

# create pandas series for production and raw materials
production = pd.Series(
    {
        "U": pyo.value(model.y_U),
        "V": pyo.value(model.y_V),
    }
)

raw_materials = pd.Series(
    {
        "A": pyo.value(model.x_A),
        "B": pyo.value(model.x_B),
        "M": pyo.value(model.x_M),
    }
)

# display pandas series
print(production)
print(raw_materials)

# Create a 1x2 grid of subplots and configure global settings
fig, ax = plt.subplots(1, 2, figsize=(8, 3.5))
plt.rcParams["font.size"] = 12
colors = plt.cm.tab20c.colors
color_sets = [[colors[0], colors[4]], [colors[16], colors[8], colors[12]]]
datasets = [production, raw_materials]
titles = ["Production", "Raw Materials"]

# Plot data on subplots
for i, (data, title, color_set) in enumerate(zip(datasets, titles, color_sets)):
    data.plot(ax=ax[i], kind="barh", title=title, alpha=0.7, color=color_set)
    ax[i].set_xlabel("Units")
    ax[i].invert_yaxis()
plt.tight_layout()
plt.show()