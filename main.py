# Código completo com lógica de alocação de drones, visualização e impressão no console

import matplotlib.pyplot as plt
import numpy as np
from functools import lru_cache
import random

# Gerar dados simulados de drones e focos de incêndio
random.seed(42)
NUM_DRONES = 4
NUM_FOCOS = 4

drones = [
    {
        "x": random.randint(0, 20),
        "y": random.randint(0, 20),
        "bateria": random.randint(60, 100),
        "capacidade": round(random.uniform(1.0, 2.5), 2)
    } for _ in range(NUM_DRONES)
]

focos = [
    {
        "x": random.randint(0, 20),
        "y": random.randint(0, 20),
        "intensidade": random.randint(1, 10),
        "vento": round(random.uniform(0.5, 1.5), 2),
        "prioridade": random.choice([1, 2])  # 1 = normal, 2 = alta
    } for _ in range(NUM_FOCOS)
]

# Função de custo
def calcula_custo(drone, foco):
    dist = np.sqrt((drone["x"] - foco["x"])**2 + (drone["y"] - foco["y"])**2)
    intensidade = foco["intensidade"]
    vento = foco["vento"]
    prioridade = foco["prioridade"]
    eficiencia = drone["capacidade"]
    custo = dist * (intensidade + vento) / eficiencia
    if prioridade == 2:
        custo *= 1.2  # penalidade se for prioridade alta
    return round(custo, 2)

# Programação dinâmica com memoização
@lru_cache(maxsize=None)
def dp(i, usados):
    if i == NUM_DRONES:
        return 0, []
    
    melhor = float('inf')
    melhor_comb = []
    
    for j in range(NUM_FOCOS):
        if not (usados >> j) & 1:
            custo = calcula_custo(drones[i], focos[j])
            if custo <= drones[i]["bateria"]:
                proximo_custo, proxima_comb = dp(i + 1, usados | (1 << j))
                total = custo + proximo_custo
                if total < melhor:
                    melhor = total
                    melhor_comb = [(i, j, custo)] + proxima_comb
    return melhor, melhor_comb

custo_total, alocacao = dp(0, 0)

# Visualização
def plot_alocacao(drones, focos, alocacao):
    plt.figure(figsize=(10, 8))
    for i, d in enumerate(drones):
        plt.scatter(d["x"], d["y"], c="blue", label="Drone" if i == 0 else "")
        plt.text(d["x"] + 0.3, d["y"], f"D{i}", fontsize=10)

    for i, f in enumerate(focos):
        cor = "red" if f["prioridade"] == 2 else "orange"
        plt.scatter(f["x"], f["y"], c=cor, label="Foco" if i == 0 else "")
        plt.text(f["x"] + 0.3, f["y"], f"F{i}", fontsize=10)

    for drone_id, foco_id, _ in alocacao:
        d = drones[drone_id]
        f = focos[foco_id]
        plt.plot([d["x"], f["x"]], [d["y"], f["y"]], c="gray", linestyle="--")

    plt.legend()
    plt.title("Alocação de Drones para Focos de Incêndio")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()

# Impressão no console
print("Alocação ótima dos drones:")
for drone_id, foco_id, custo in alocacao:
    print(f"Drone {drone_id} → Foco {foco_id} | Custo: {custo}")

print(f"\nCusto total da operação: {custo_total}")

print("\nDrones:")
for i, d in enumerate(drones):
    print(f"Drone {i}: Posição=({d['x']},{d['y']}), Bateria={d['bateria']}, Capacidade={d['capacidade']:.2f}")

print("\nFocos:")
for i, f in enumerate(focos):
    prioridade = "Alta" if f["prioridade"] == 2 else "Normal"
    print(f"Foco {i}: Posição=({f['x']},{f['y']}), Intensidade={f['intensidade']}, Vento={f['vento']:.2f}, Prioridade={prioridade}")

# Exibir gráfico
plot_alocacao(drones, focos, alocacao)
