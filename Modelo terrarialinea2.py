import pandas as pd
import matplotlib.pyplot as plt
import pyomo.environ as pyo
from pyomo.contrib.iis import write_iis
solver = 'appsi_highs'
SOLVER = pyo.SolverFactory(solver)
assert SOLVER.available(), f"Solver {solver} is not available."
""" First step """
# read data from file into pandas DataFrame

asasassa
asasas
asdadsdasasd
asasdadasdasads

db = pd.read_csv("Archivo_NPC_Biomacopia.txt", index_col="NPCs", skipinitialspace=True)
print(db)

nn = pd.read_csv("Archivo_NPC_NPCcopia.txt", index_col="NPCs", skipinitialspace=True)
print(nn)

db.index =db.index.str.strip()
db.columns = db.columns.str.strip()

nn.index =nn.index.str.strip()
nn.columns = nn.columns.str.strip()
""" Second step """
# create model with optional problem title
model = pyo.ConcreteModel("Asignacion de NPCs a biomas")
# display model
model.display()
# create index sets for NPCs and biomas
model.NPCS   = pyo.Set(initialize=list(db.index))
model.BIOMAS = pyo.Set(initialize=list(db.columns))
model.NPCS1  = pyo.Set(initialize=list(nn.index))
model.NPCS2  = pyo.Set(initialize=list(nn.columns)) 

"""" third step """
# create decision variables
model.X_nb = pyo.Var(model.NPCS, model.BIOMAS, domain=pyo.Binary)
model.Y_b  = pyo.Var(model.BIOMAS, domain=pyo.NonNegativeIntegers)
model.X_nn = pyo.Var(model.NPCS1, model.NPCS2, domain=pyo.Binary)
model.Z_nb = pyo.Var(model.NPCS1, model.NPCS2, model.BIOMAS, domain=pyo.Binary) #Vble Auxiliar para relacionar las variables de asignación entre NPCs y biomas con las relaciones entre NPCs
model.Y_bn = pyo.Var(model.BIOMAS, domain=pyo.Binary)
# display updated model
model.display()

""" fourth step """
# create expressions 

afinidadb_data = {(n, b): float(db.loc[n, b]) for n in db.index for b in db.columns}
model.afinidadb = pyo.Param(model.NPCS, model.BIOMAS, initialize=afinidadb_data, within=pyo.NonNegativeReals)

afinidadn_data = {(n, b): float(nn.loc[n, b]) for n in nn.index for b in nn.columns}
model.afinidadn = pyo.Param(model.NPCS1, model.NPCS2, initialize=afinidadn_data, within=pyo.NonNegativeReals)

# expressions can be printed
print(model.afinidadb)
print(model.afinidadn)


model.afinidad_b = pyo.Expression(expr = sum(model.afinidadb[n, b] * model.X_nb[n, b] for n in model.NPCS for b in model.BIOMAS))
model.afinidad_n = pyo.Expression(expr = sum(model.afinidadn[n1, n2] * model.Z_nb[n1, n2, b] for n1 in model.NPCS1 for n2 in model.NPCS2 for b in model.BIOMAS if n1 != n2))

""" fifth step """
# create objective function
model.Afinidad_T = pyo.Objective(expr=model.afinidad_b + model.afinidad_n, sense=pyo.minimize)


""" Sixth step """
# create constraints Logical realations between variables
print(model.Afinidad_T)
model.asignacion_unica = pyo.Constraint(model.NPCS)   
for n in model.NPCS:
    model.asignacion_unica[n] = sum(model.X_nb[n, b] for b in model.BIOMAS) == 1

model.conteo_bioma = pyo.Constraint(model.BIOMAS)
for b in model.BIOMAS:
    model.conteo_bioma[b] = model.Y_b[b] == sum(model.X_nb[n, b] for n in model.NPCS) # aux

model.conteo_aux = pyo.Constraint(model.NPCS, model.BIOMAS)
for n in model.NPCS:
    for b in model.BIOMAS:
        model.conteo_aux[n, b] = model.Y_b[b] >= model.X_nb[n, b]  # 10s

model.capacidad = pyo.Constraint(model.BIOMAS)
model.arbitraria3 = pyo.Constraint(model.BIOMAS)
model.capacidadMIN = pyo.Constraint(model.BIOMAS)
for b in model.BIOMAS:
    model.capacidad[b] = model.Y_b[b] <= 3 * model.Y_bn[b]
    model.capacidadMIN[b] = model.Y_b[b] >= 2 * model.Y_bn[b]
    model.arbitraria3[b]  = model.Y_b[b] >= model.Y_bn[b]  # BUENA 82 - 95 S A 62 -72 S

model.vecinosaux0 = pyo.Constraint(model.NPCS1, model.NPCS2, model.BIOMAS)
model.vecinosaux1 = pyo.Constraint(model.NPCS1, model.NPCS2, model.BIOMAS)
model.vecinosaux2 = pyo.Constraint(model.NPCS1, model.NPCS2, model.BIOMAS)
model.arbitraria1 = pyo.Constraint(model.NPCS1, model.NPCS2, model.BIOMAS) #duda posibles 5 -10s     
model.arbitraria2 = pyo.Constraint(model.NPCS1, model.NPCS2, model.BIOMAS) # duda posibles 5 -10s5
model.arbitraria5 = pyo.Constraint(model.NPCS1, model.NPCS2, model.BIOMAS) #duda posibles 5 -10s
for n1 in model.NPCS1:
    for n2 in model.NPCS2:
        if n1 != n2:
            for b in model.BIOMAS:
                model.vecinosaux0[n1, n2, b] = model.Z_nb[n1, n2, b] <= model.X_nb[n1, b]
                model.vecinosaux1[n1, n2, b] = model.Z_nb[n1, n2, b] <= model.X_nb[n2, b]
                model.vecinosaux2[n1, n2, b] = model.Z_nb[n1, n2, b] >= model.X_nb[n1, b] + model.X_nb[n2, b] - 1
                model.arbitraria1[n1, n2, b] = model.Z_nb[n2, n1, b] <= model.X_nb[n2, b]
                model.arbitraria2[n1, n2, b] = model.Z_nb[n2, n1, b] <= model.X_nb[n2, b]
                model.arbitraria5[n1, n2, b] = model.Z_nb[n2, n1, b] >= model.X_nb[n2, b] + model.X_nb[n1, b] - 1

""" Seventh step """
# Solve the model using a solver
results = SOLVER.solve(model, tee=True)

# # --- BLOQUE IIS AÑADIDO ---
if results.solver.termination_condition == pyo.TerminationCondition.infeasible:
    write_iis(model, iis_file_name="infeasible_model.mps", solver='appsi_highs')

"""Eighth step"""
print(f"\n{'REPORTE DE OPTIMIZACIÓN DE VECINDARIO':^60}")

for b in model.BIOMAS:
    if pyo.value(model.Y_b[b]) > 0.1:
        print(f"\n BIOMA: {b.upper()}")
        poblacion = int(pyo.value(model.Y_b[b]))
        print(f"Población: {poblacion} NPC(s)")
            
        z0_bioma = 0
        npcs_en_bioma = [n for n in model.NPCS if pyo.value(model.X_nb[n, b]) > 0.5]
        
        # 1. Mostrar información detallada por NPC con su vecindad
        for n in npcs_en_bioma:
            afinidad_base = float(db.loc[n, b])
            z0_bioma += afinidad_base
            
            # Buscamos los factores de vecindad que recibe este NPC de los otros en el bioma
            factores_vecinos = []
            for vecino in npcs_en_bioma:
                if n != vecino:
                    try:
                        f_vecino = float(nn.loc[n, vecino])
                        factores_vecinos.append(f"{f_vecino:.2f}(con {vecino})")
                    except KeyError:
                        factores_vecinos.append(f"1.00(con {vecino})")
            
            vecindad_str = " | Vecindad: " + ", ".join(factores_vecinos) if factores_vecinos else " | Sin vecinos"
            print(f"  • {n:15s} | Afinidad Bioma: {afinidad_base:.2f}{vecindad_str}")     
    
        print("\n  Cálculo Real (Multiplicativo):")
        for n1 in npcs_en_bioma:
            factor_bioma_n1 = float(db.loc[n1, b])
            felicidad_real_n1 = factor_bioma_n1
            formulas_vecinos = []

            for n2 in npcs_en_bioma:
                if n1 != n2:
                    try:
                        bonus = float(nn.loc[n1, n2])
                    except KeyError:
                        bonus = 1.00
                    
                    felicidad_real_n1 *= bonus
                    z0_bioma += bonus 
                    formulas_vecinos.append(f"{bonus:.2f}")

            if formulas_vecinos:
                formula_txt = f"{factor_bioma_n1:.2f} * {' * '.join(formulas_vecinos)}"
                print(f"    - Real {n1:10s}: {formula_txt} = {felicidad_real_n1:.3f}")
            else:
                print(f"    - Real {n1:10s}: {factor_bioma_n1:.2f} = {felicidad_real_n1:.3f}")
            
    valor_objetivo = pyo.value(model.Afinidad_T)
    print(f"AFINIDAD TOTAL FINAL (Suma de Z0): {valor_objetivo:.2f}")

