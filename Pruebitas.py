# -*- coding: utf-8 -*-
import pandas as pd
import pyomo.environ as pyo

# ========== 0) Utilidades ==========
def solve_with_fallback(model):
    """
    Intenta resolver con 'highs' (wrapper estable), luego 'glpk', y por último 'appsi_highs' (sin tee).
    Devuelve (results, solver_name_usado).
    """
    for solver_name, tee in [('highs', True), ('glpk', True), ('appsi_highs', False)]:
        try:
            solver = pyo.SolverFactory(solver_name)
            if solver is None or not solver.available():
                continue
            print(f"\n--> Probando solver: {solver_name} (tee={tee})")
            res = solver.solve(model, tee=tee)
            print(f"✔ Solver '{solver_name}' finalizó con estado: {res.solver.status}, "
                  f"terminación: {res.solver.termination_condition}")
            return res, solver_name
        except Exception as e:
            print(f"✖ Solver '{solver_name}' falló: {e}")
            continue
    raise RuntimeError("No se pudo resolver el modelo con highs/glpk/appsi_highs. "
                       "Instala alguno o verifica tu PATH.")

def pares_no_dirigidos(lista_npcs):
    L = list(lista_npcs)
    return [(L[i], L[j]) for i in range(len(L)) for j in range(i+1, len(L))]


# ========== 1) Cargar y sanear datos ==========
db = pd.read_csv("Archivo_NPC_Bioma.txt", index_col="NPCs", skipinitialspace=True)
nn = pd.read_csv("Archivo_NPC_NPC.txt", index_col="NPCs", skipinitialspace=True)

# Limpieza de espacios
db.index = db.index.str.strip()
db.columns = db.columns.str.strip()
nn.index = nn.index.str.strip()
nn.columns = nn.columns.str.strip()

# Usar la intersección de NPCs presentes en ambos archivos (filtra filas/columnas)
npcs_db = set(db.index)
npcs_nn = set(nn.index) & set(nn.columns)
npcs_comunes = sorted(list(npcs_db & npcs_nn))
if len(npcs_comunes) < len(npcs_db) or len(npcs_comunes) < len(nn.index):
    print("⚠️ Aviso: Se detectaron NPCs no comunes entre archivos. Se usará la intersección.")
db = db.loc[npcs_comunes, :]
nn = nn.loc[npcs_comunes, npcs_comunes]

# Validación básica
assert db.notna().all().all(), "Hay NaNs en Archivo_NPC_Bioma.txt."
assert nn.notna().all().all(), "Hay NaNs en Archivo_NPC_NPC.txt."

NPCS = list(db.index)
BIOMAS = list(db.columns)
PARES = [(NPCS[i], NPCS[j]) for i in range(len(NPCS)) for j in range(i+1, len(NPCS))]

# ========== 2) Modelo ==========
model = pyo.ConcreteModel("Asignacion de NPCs a biomas (AB*AP)")

# Conjuntos
model.NPCS   = pyo.Set(initialize=NPCS)
model.BIOMAS = pyo.Set(initialize=BIOMAS)
model.PARES  = pyo.Set(initialize=PARES, dimen=2)

# Parámetros (no imponemos [0,1], solo no-negativos; si tus datos son porcentajes, normalízalos antes)
afinidad_bioma = {(n, b): float(db.loc[n, b]) for n in NPCS for b in BIOMAS}
afinidad_par   = {(i, j): float(nn.loc[i, j]) for (i, j) in PARES}

model.AB = pyo.Param(model.NPCS, model.BIOMAS, initialize=afinidad_bioma, within=pyo.NonNegativeReals)
model.AP = pyo.Param(model.PARES, initialize=afinidad_par,   within=pyo.NonNegativeReals)

# ========== 3) Variables ==========
# X_nb[n,b] = 1 si el NPC n está en bioma b
model.X_nb = pyo.Var(model.NPCS, model.BIOMAS, domain=pyo.Binary)
# Z_nb[i,j,b] = 1 si el par (i,j) convive en el bioma b (i<j)
model.Z_nb = pyo.Var(model.PARES, model.BIOMAS, domain=pyo.Binary)

# ========== 4) Restricciones ==========
# Asignación única por NPC
def asignacion_unica_rule(m, n):
    return sum(m.X_nb[n, b] for b in m.BIOMAS) == 1
model.asignacion_unica = pyo.Constraint(model.NPCS, rule=asignacion_unica_rule)

# Capacidad por bioma (máximo 3 NPCs): sum_n X_nb[n,b] <= 3
def capacidad_rule(m, b):
    return sum(m.X_nb[n, b] for n in m.NPCS) <= 3
model.capacidad = pyo.Constraint(model.BIOMAS, rule=capacidad_rule)

# Linealización de conjunción: Z_ijb = 1  <=>  X_ib = X_jb = 1
def vecinos_le1(m, i, j, b): return m.Z_nb[i, j, b] <= m.X_nb[i, b]
def vecinos_le2(m, i, j, b): return m.Z_nb[i, j, b] <= m.X_nb[j, b]
def vecinos_ge(m, i, j, b):  return m.Z_nb[i, j, b] >= m.X_nb[i, b] + m.X_nb[j, b] - 1
model.vecinos_le1 = pyo.Constraint(model.PARES, model.BIOMAS, rule=vecinos_le1)
model.vecinos_le2 = pyo.Constraint(model.PARES, model.BIOMAS, rule=vecinos_le2)
model.vecinos_ge  = pyo.Constraint(model.PARES, model.BIOMAS, rule=vecinos_ge)

# (Opcional) Si quieres forzar convivencia (evitar objetivo 0 por no crear pares),
# descomenta esta cota inferior de población: sum_n X_nb[n,b] >= 2
# def min_poblacion_rule(m, b):
#     return sum(m.X_nb[n, b] for n in m.NPCS) >= 2
# model.min_poblacion = pyo.Constraint(model.BIOMAS, rule=min_poblacion_rule)

# ========== 5) Objetivo ==========
# Minimizamos: Σ_b Σ_(i<j) [ (AB[i,b] + AB[j,b]) * AP[i,j] * Z[i,j,b] ]
def objetivo_rule(m):
    return sum((m.AB[i, b] + m.AB[j, b]) * m.AP[i, j] * m.Z_nb[i, j, b]
               for (i, j) in m.PARES for b in m.BIOMAS)
model.Afinidad_T = pyo.Objective(rule=objetivo_rule, sense=pyo.minimize)

# ========== 6) Resolver con fallback ==========
results, used_solver = solve_with_fallback(model)

# ========== 7) Reporte ==========
print(f"\n{'REPORTE DE OPTIMIZACIÓN (AB*AP)':^72}")

suma_global = 0.0
for b in model.BIOMAS:
    # NPCs asignados a este bioma
    npcs_en_bioma = [n for n in model.NPCS if pyo.value(model.X_nb[n, b]) > 0.5]
    pob = len(npcs_en_bioma)
    if pob > 0:
        print(f"\n BIOMA: {b.upper()}")
        print(f"Población: {pob} NPC(s)")
        for n in npcs_en_bioma:
            ab = float(pyo.value(model.AB[n, b]))
            print(f"  • {n:15s} | AB: {ab:.4f}")

        print("  Interacciones (pares activos):")
        encontro = False
        subtotal_b = 0.0
        for (i, j) in pares_no_dirigidos(npcs_en_bioma):
            if pyo.value(model.Z_nb[i, j, b]) > 0.5:
                ab_i = float(pyo.value(model.AB[i, b]))
                ab_j = float(pyo.value(model.AB[j, b]))
                ap_ij = float(pyo.value(model.AP[i, j]))
                contrib = (ab_i + ab_j) * ap_ij
                print(f"    - {i} - {j} | (AB_i+AB_j)*AP = ({ab_i:.4f}+{ab_j:.4f})*{ap_ij:.4f} = {contrib:.4f}")
                subtotal_b += contrib
                encontro = True
        if not encontro:
            print("    - No hay pares activos en este bioma.")

        print(f"➤ SUBTOTAL (Σ productos) en {b.upper()}: {subtotal_b:.6f}")
        suma_global += subtotal_b

valor_obj = float(pyo.value(model.Afinidad_T))
print(f"\nOBJETIVO GLOBAL (mínimo de Σ (AB*AP) por pares activos): {valor_obj:.6f}")
print(f"Suma de subtotales por bioma:                         {suma_global:.6f}")
print(f"Solver usado: {used_solver}")