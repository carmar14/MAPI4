from lnn import LiquidTank
import torch
from componentes import Tanque
from simulacion import SimulacionNivel
import numpy as np
import json

# --- 1. Configuración de entrada (Escalones) ---
def generar_escalones_random(n_segmentos=50, duracion=300):
    valores = np.random.uniform(0.5, 2.0, n_segmentos)
    return np.repeat(valores, duracion)

q_in_signal = generar_escalones_random(180, 400)
#q_in_signal = np.repeat([0.8, 1.2, 0.8, 2.0, 1.5], 700) # Amplitudes variables

# --- 2. Crear objetos ---
mi_tanque = Tanque(A=2.5, a=1)
sim = SimulacionNivel(modelo_tanque=mi_tanque, dt=0.01, h0=0.0)

# --- 3. Resultados simulacion ---
time, niveles = sim.ejecutar(q_in_signal)
h = niveles
h_state = torch.zeros(1, 20)
predictions = []

model = LiquidTank(2, 20)
model.load_state_dict(torch.load("best_lnn_model.pth"))
model.eval()

for i in range(len(time)-1):
    q = torch.tensor([[q_in_signal[i], h[i]]], dtype=torch.float32)
    pred, h_state = model(q, h_state)
    predictions.append(pred.item())

# Convertir todo a listas estándar
results = {
    "time": time[1:].tolist(),
    "h_real": h[1:].tolist(),
    "h_pred": predictions
}

# Guardar archivo
with open("resultados_lnn_.json", "w") as f:
    json.dump(results, f, indent=4)

print("Resultados guardados en resultados_lnn.json")