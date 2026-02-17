
""" First step import Pyomo library and declare solver """
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
model.x_P = pyo.Var(bounds=(0, 1750))
model.x_S = pyo.Var(bounds=(0, 1000))
model.x_C = pyo.Var(bounds=(0, 4800))
model.x_G = pyo.Var(bounds=(0, 1500))

model.y_L = pyo.Var(bounds=(0, None))
model.y_M = pyo.Var(bounds=(0, None))
 
# display updated model
model.display()
""" fifth step """
# create objective function
model.profit = pyo.Objective(expr=12 * model.y_L + 9 * model.y_M, sense=pyo.maximize)

""" Sixth step """
# create constraints Logical realations between variables
model.raw_materials = pyo.Constraint(expr = 4 * model.y_L + 2 * model.y_M <= model.x_C)
model.labor_A = pyo.Constraint(expr = model.y_M <= model.x_G)
model.labor_B = pyo.Constraint(expr = model.y_L <= model.x_S)
model.labor_C = pyo.Constraint(expr = model.y_L + model.y_M <= model.x_P)


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

print(f"L (lógicos) = {pyo.value(model.y_L):.0f}")
print(f"M (memoria) = {pyo.value(model.y_M):.0f}")
print(f" Profit = {pyo.value(model.profit): 9.2f}")

print("x_P =", model.x_P())
print("x_S =", model.x_S())
print("x_C =", model.x_C())
print("x_G =", model.x_G())