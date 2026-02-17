import pandas as pd
import matplotlib.pyplot as plt
import pyomo.environ as pyo
import numpy as np
solver = 'appsi_highs'
SOLVER = pyo.SolverFactory(solver)
assert SOLVER.available(), f"Solver {solver} is not available."
""" First step """
# read data from file into pandas DataFrame

db = pd.read_csv("Archivo_NPC_Bioma.txt", index_col="NPCs", skipinitialspace=True)
print(db)

nn = pd.read_csv("Archivo_NPC_NPC.txt", index_col="NPCs", skipinitialspace=True)
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
model.Z_0  = pyo.Var(model.NPCS, domain=pyo.NonNegativeReals)

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
model.Afinidad_T = pyo.Objective(expr=sum(model.Z_0[n] for n in model.NPCS), sense=pyo.minimize)
""" Sixth step """
# create constraints Logical realations between variables
print(model.Afinidad_T)
model.asignacion_unica = pyo.Constraint(model.NPCS)
for n in model.NPCS:
    model.asignacion_unica[n] = sum(model.X_nb[n, b] for b in model.BIOMAS) == 1

model.conteo_bioma = pyo.Constraint(model.BIOMAS)
for b in model.BIOMAS:
    model.conteo_bioma[b] = model.Y_b[b] == sum(model.X_nb[n, b] for n in model.NPCS)

model.capacidad = pyo.Constraint(model.BIOMAS)
for b in model.BIOMAS:
    model.capacidad[b] = model.Y_b[b] <= 3

model.vecinos = pyo.Constraint(model.NPCS1, model.NPCS2, model.BIOMAS)
for n1 in model.NPCS1:
    for n2 in model.NPCS2:
        if n1 != n2:
            for b in model.BIOMAS:
                model.vecinos[n1, n2, b] = model.Z_nb[n1, n2, b] <= model.X_nb[n1, b]
                model.vecinos[n1, n2, b] = model.Z_nb[n1, n2, b] <= model.X_nb[n2, b]
                model.vecinos[n1, n2, b] = model.Z_nb[n1, n2, b] >= model.X_nb[n1, b] + model.X_nb[n2, b] - 1


# Nueva restricción para "anclar" Z_0 al valor real de afinidad
model.vincular_Z0 = pyo.Constraint(model.NPCS)

for n in model.NPCS:
    # Afinidad del bioma donde está el NPC
    term_bioma = sum(model.afinidadb[n, b] * model.X_nb[n, b] for b in model.BIOMAS)
    
    # Afinidad de los vecinos cercanos en el mismo bioma
    term_vecinos = sum(model.afinidadn[n, n2] * model.Z_nb[n, n2, b] 
                       for n2 in model.NPCS2 for b in model.BIOMAS if n != n2)
    
    # Z_0 debe ser al menos la suma de estas afinidades (para que al minimizar no sea 0)
    model.vincular_Z0[n] = model.Z_0[n] >= term_bioma + term_vecinos



# Definimos dos conjuntos de restricciones para cumplir l1*x2 <= z <= u1*x2
model.linear_prod_LB = pyo.Constraint(model.NPCS)
model.linear_prod_UB = pyo.Constraint(model.NPCS)

l1, u1 = 0.1, 20 
for n in model.NPCS:
    # x2: la suma de afinidades por vecinos en el bioma actual
    x2 = sum(model.afinidadn[n, n2] * model.Z_nb[n, n2, b] 
             for n2 in model.NPCS2 for b in model.BIOMAS if n != n2)
    # Restricción Límite Inferior: l1 * x2 <= z
    model.linear_prod_LB[n] = l1 * x2 <= model.Z_0[n]
    # Restricción Límite Superior: z <= u1 * x2
    model.linear_prod_UB[n] = model.Z_0[n] <= u1 * x2

""" Seventh step """
# Solve the model using a solver
results = SOLVER.solve(model, tee=True)

# """Eighth step"""
# # display results and report the solution
# print(f"{'REPORTE DE OPTIMIZACIÓN DE VECINDARIO':^60}")

# for b in model.BIOMAS:
#     # Solo mostrar biomas que tengan al menos un NPC
#     if pyo.value(model.Y_b[b]) > 0.1:
#         print(f"\n BIOMA: {b.upper()}")
#         print(f"Población: {int(pyo.value(model.Y_b[b]))} NPC(s)")
            
#         npcs_en_bioma = []
#         for n in model.NPCS:
#             if pyo.value(model.X_nb[n, b]) > 0.5:
#                     afinidad_base = db.loc[n, b] 
#                     print(f"  • {n:15s} | Afinidad Bioma: {afinidad_base:.2f}") 
#                     npcs_en_bioma.append(n)
            
#         print("  Interacciones:")
#         encontro_vecinos = False
#         for n1 in npcs_en_bioma:
#             for n2 in npcs_en_bioma:
#                 if n1 != n2:
#                     if pyo.value(model.Z_nb[n1, n2, b]) > 0.5:
#                         bonus = model.afinidadn[n1, n2]
#                         print(f"    - {n1} - {n2} (Factor: {bonus:.2f})")
#                         encontro_vecinos = True
            
#         if not encontro_vecinos and len(npcs_en_bioma) > 1:
#                 print("    - Sin bonos de vecindad activos.")
#     try:
#         valor_objetivo = pyo.value(model.Afinidad_T)
#         print(f"AFINIDAD TOTAL (Puntaje de felicidad): {valor_objetivo:.2f}")
#     except:
#         print("No se pudo calcular el valor del objetivo.")

from pyomo.opt import TerminationCondition
"""Eighth step"""
if results.solver.termination_condition == TerminationCondition.optimal:
    print("\n" + "="*70)
    print(f"{'REPORTE DE OPTIMIZACIÓN TERRARIA (Linealizado)':^70}")
    print("="*70)

    for b in model.BIOMAS:
        if pyo.value(model.Y_b[b]) > 0.5:
            print(f"\n🏠 BIOMA: {b.upper()} (Población: {int(pyo.value(model.Y_b[b]))})")
            print("-" * 70)
            print(f"{'NPC':<15} | {'Afin. Bioma':<12} | {'Afin. Vecinos':<15} | {'Total Z0':<10}")
            print("-" * 70)
            
            for n in model.NPCS:
                if pyo.value(model.X_nb[n, b]) > 0.5:
                    # Cálculo de indicadores para el display
                    val_bioma = db.loc[n, b]
                    val_vecinos = sum(nn.loc[n, n2] for n2 in model.NPCS2 
                                    if n != n2 and pyo.value(model.Z_nb[n, n2, b]) > 0.5)
                    val_total = pyo.value(model.Z_0[n])
                    
                    print(f"{n:<15} | {val_bioma:<12.2f} | {val_vecinos:<15.2f} | {val_total:<10.2f}")
            
            # Mostrar relaciones específicas
            for n1 in model.NPCS:
                for n2 in model.NPCS2:
                    if n1 != n2 and pyo.value(model.Z_nb[n1, n2, b]) > 0.5:
                        print(f"   [Relación: {n1} + {n2} -> Bono: {nn.loc[n1, n2]:.2f}]")

    print("\n" + "="*70)
    print(f"VALOR DE LA FUNCIÓN OBJETIVO (Suma Z0): {pyo.value(model.Afinidad_T):.2f}")
    print("="*70)