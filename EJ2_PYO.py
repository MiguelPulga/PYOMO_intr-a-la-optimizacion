from io import StringIO
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pyomo.environ as pyo
solver = 'appsi_highs'
SOLVER = pyo.SolverFactory(solver)
assert SOLVER.available(), f"Solver {solver} is not available."

demand_data = """
chip, Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec
logic, 88, 125, 260, 217, 238, 286, 248, 238, 265, 293, 259, 244
memory, 47, 62, 81, 65, 95, 118, 86, 89, 82, 82, 84, 66
"""

demand_chips = pd.read_csv(StringIO(demand_data), index_col="chip")
print(demand_chips)

use = dict()
use["logic"] = {"silicon": 1, "plastic": 1, "copper": 4}
use["memory"] = {"germanium": 1, "plastic": 1, "copper": 2}
use = pd.DataFrame.from_dict(use).fillna(0).astype(int)
print(use)

demand = use.dot(demand_chips)
print(demand)   
products = {
    "U": {"price": 270, "demand": 40},
    "V": {"price": 210, "demand": None},
}

resources = {
    "M": {"price": 10, "available": None},
    "labor A": {"price": 50, "available": 80},
    "labor B": {"price": 40, "available": 100},
}

processes = {
    "U": {"M": 10, "labor A": 1, "labor B": 2},
    "V": {"M": 9, "labor A": 1, "labor B": 1},
}

for product, process in processes.items():
    for resource, value in process.items():
        print(f"{product:4s} {resource:10s} {value}")

for resource, attributes in resources.items():
    for attribute, value in attributes.items():
        print(f"{resource:8s} {attribute:10s} {value}")

# print data

for product, attributes in products.items():
    for attribute, value in attributes.items():
        print(f"{product} {attribute:10s} {value}")

""" Second step """
# create model with optional problem title
model = pyo.ConcreteModel("Production Planning: Version 2")
# display model
model.display()

"""" third step """
model.PRODUCTS = pyo.Set(initialize=products.keys())
model.RESOURCES = pyo.Set(initialize=resources.keys())

# display updated model
model.display()

# parameter for bounds
@model.Param(model.PRODUCTS, domain=pyo.Any)
def demand(model, product):
    return products[product]["demand"]


@model.Param(model.RESOURCES, domain=pyo.Any)
def available(model, resource):
    return resources[resource]["available"]


# parameter with price coefficients
@model.Param(model.PRODUCTS)
def cp(model, product):
    return products[product]["price"]


@model.Param(model.RESOURCES)
def cr(model, resource):
    return resources[resource]["price"]


# process parameters: a[r,p]
@model.Param(model.RESOURCES, model.PRODUCTS)
def a(model, resource, product):
    return processes[product][resource]

model.x = pyo.Var(
    model.RESOURCES, bounds=lambda model, resource: (0, model.available[resource])
)
model.y = pyo.Var(
    model.PRODUCTS, bounds=lambda model, product: (0, model.demand[product])
)

model.revenue = pyo.quicksum(
    model.cp[product] * model.y[product] for product in model.PRODUCTS
)
model.cost = pyo.quicksum(
    model.cr[resource] * model.x[resource] for resource in model.RESOURCES
)


# create objective
@model.Objective(sense=pyo.maximize)
def profit(model):
    return model.revenue - model.cost

# create indexed constraint
@model.Constraint(model.RESOURCES)
def materials_used(model, resource):
    return (
        pyo.quicksum(
            model.a[resource, product] * model.y[product] for product in model.PRODUCTS
        )
        <= model.x[resource]
    )

model.pprint()

# solve
SOLVER.solve(model)

# create a solution report
print(f"Profit = {pyo.value(model.profit)}")

print("\nProduction Report")
for product in model.PRODUCTS:
    print(f" {product}  produced =  {pyo.value(model.y[product])}")

print("\nResource Report")
for resource in model.RESOURCES:
    print(f" {resource} consumed = {pyo.value(model.x[resource])}")