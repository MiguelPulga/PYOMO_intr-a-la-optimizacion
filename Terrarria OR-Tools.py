from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model
import pandas as pd
import matplotlib.pyplot as plt

db = pd.read_csv("Archivo_NPC_Bioma.txt", index_col="NPCs", skipinitialspace=True)
print(db)

nn = pd.read_csv("Archivo_NPC_NPC.txt", index_col="NPCs", skipinitialspace=True)
print(nn)

for df in [db, nn]:
    df.index = df.index.str.strip()
    df.columns = df.columns.str.strip()

# Transformación a enteros (VALOR - 1) * 100
db_int = ((db - 1) * 100).round().astype(int)
nn_int = ((nn - 1) * 100).round().astype(int)

NPCS = list(db.index)
BIOMAS = list(db.columns)
NPCS1 = list(nn.index)
NPCS2 = list(nn.columns)

# --- 2. Inicializar Modelo ---
model = cp_model.CpModel()

# --- 3. Variables de Decisión ---
# X_nb: NPC 'n' está en bioma 'b'
X_nb = {(n, b): model.NewBoolVar(f'X_nb_{n}_{b}') for n in NPCS for b in BIOMAS}

# Y_b: Conteo de NPCs en bioma 'b'
Y_b = {b: model.NewIntVar(0, len(NPCS), f'Y_b_{b}') for b in BIOMAS}

# Y_bn: Interruptor binario (1 si el bioma 'b' está activo)
Y_bn = {b: model.NewBoolVar(f'Y_bn_{b}') for b in BIOMAS}

# X_nn: Binaria que indica si n1 y n2 están en el MISMO bioma (cualquiera)
X_nn = {(n1, n2): model.NewBoolVar(f'X_nn_{n1}_{n2}') for n1 in NPCS1 for n2 in NPCS2}

# Z_nb: Auxiliar (n1 y n2 están juntos específicamente en el bioma 'b')
Z_nb = {(n1, n2, b): model.NewBoolVar(f'Z_nb_{n1}_{n2}_{b}') for n1 in NPCS1 for n2 in NPCS2 for b in BIOMAS}

# --- 4. Restricciones (Constraints) ---

# A. Asignación única por NPC
for n in NPCS:
    model.Add(sum(X_nb[n, b] for b in BIOMAS) >= 1)

# B. Lógica de Biomas (Conteo y Rango 2-3)
for b in BIOMAS:
    model.Add(Y_b[b] == sum(X_nb[n, b] for n in NPCS))
    # Restricción Semicontinua (Rango o Cero)
    model.Add(Y_b[b] <= 3 * Y_bn[b])
    model.Add(Y_b[b] >= 2 * Y_bn[b])
    model.Add(Y_b[b] >= Y_bn[b]) # aux 1
    for n in NPCS:
        model.Add(Y_b[b] >= X_nb[n, b]) #aux 2

# C. Lógica para Z_nb (n1 y n2 juntos en el bioma b)
for n1 in NPCS1:
    for n2 in NPCS2:
        if n1 != n2:
            for b in BIOMAS:
                # Z_nb[n1,n2,b] es verdadero SSI X_nb[n1,b] Y X_nb[n2,b] son 1
                model.Add(Z_nb[n1, n2, b] <= X_nb[n1, b])
                model.Add(Z_nb[n1, n2, b] <= X_nb[n2, b])
                model.Add(Z_nb[n1, n2, b] >= X_nb[n1, b] + X_nb[n2, b] - 1)

# D. Lógica para X_nn (n1 y n2 están juntos en ALGÚN bioma)
for n1 in NPCS1:
    for n2 in NPCS2:
        if n1 != n2:
            # X_nn es la suma de las Z_nb para todos los biomas
            # Como un NPC solo está en un bioma, máximo un Z_nb será 1
            model.Add(X_nn[n1, n2] == sum(Z_nb[n1, n2, b] for b in BIOMAS))

# --- 5. Función Objetivo ---
# Maximizamos afinidad bioma + afinidad vecinos (usando valores enteros)
obj_bioma = sum(db_int.loc[n, b] * X_nb[n, b] for n in NPCS for b in BIOMAS)
obj_vecinos = sum(nn_int.loc[n1, n2] * X_nn[n1, n2] for n1 in NPCS1 for n2 in NPCS2 if n1 != n2)

model.Minimize(obj_bioma + obj_vecinos) # Cambiar a Maximize si prefieres afinidad alta

# --- 6. Resolución ---
solver = cp_model.CpSolver()
status = solver.Solve(model)
# Esto imprimirá el valor de la función objetivo en la nueva escala entera
print(f"Valor objetivo en escala entera: {solver.ObjectiveValue()}")

"""Eighth step - Visualización con Multiplicación Total de Bioma"""
print(f"\n{'REPORTE DE OPTIMIZACIÓN DE VECINDARIO':^75}")

total_afinidad_original = 0 

for b in BIOMAS:
    if solver.Value(Y_b[b]) > 0.1:
        print(f" BIOMA: {b.upper()}")
        poblacion = int(solver.Value(Y_b[b]))
        print(f" Población: {poblacion} NPC(s)")
            
        npcs_en_bioma = [n for n in NPCS if solver.Value(X_nb[n, b]) > 0.5]
        
        z0_bioma_acumulada = 0
        producto_bioma = 1.0  # Variable para la multiplicación total del bioma
        
        print(f" {'Nombre':<25} | {'Afinidad':<10} | {'Vecindad'}")

        for n in npcs_en_bioma:
            afinidad_base = float(db.loc[n, b])
            z0_npc = afinidad_base 
            
            factores_vecinos = []
            for vecino in npcs_en_bioma:
                if n != vecino:
                    try:
                        f_vecino = float(nn.loc[n, vecino])
                        z0_npc += (f_vecino - 1) 
                        factores_vecinos.append(f"{f_vecino:.2f}({vecino})")
                    except KeyError:
                        factores_vecinos.append(f"1.00({vecino})")
            
            z0_bioma_acumulada += z0_npc
            vecindad_str = ", ".join(factores_vecinos) if factores_vecinos else "Sin vecinos"
            print(f" • {n:<24} | {afinidad_base:<10.2f} | {vecindad_str}")     
    
        print(f"\n Suma Z0 del Bioma (Suma de factores): {z0_bioma_acumulada:.2f}")
        print(f" Cálculo Real Multiplicativo:")
        
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
                    formulas_vecinos.append(f"{bonus:.2f}")

            # Acumulamos el producto total del bioma
            producto_bioma *= felicidad_real_n1
            total_afinidad_original += felicidad_real_n1
            
            formula_txt = f"{factor_bioma_n1:.2f} * {' * '.join(formulas_vecinos)}" if formulas_vecinos else f"{factor_bioma_n1:.2f}"
            print(f"   - Real {n1:<18}: {formula_txt:<20} = {felicidad_real_n1:.3f}")

        print(f" >>> MULTIPLICACIÓN TOTAL DEL BIOMA: {producto_bioma:.4f}")

print(f"AFINIDAD TOTAL FINAL (Suma de felicidades): {total_afinidad_original:.2f}")


