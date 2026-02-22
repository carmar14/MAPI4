import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from lnn import LiquidTank
import torch.optim as optim
import torch.nn as nn
import torch
from componentes import Tanque
from simulacion import SimulacionNivel
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Configuración de entrada (Escalones) ---
q_in_signal = np.repeat([0.8, 1.2, 0.8, 2.0, 1.5], 700) # Amplitudes variables

# --- 2. Crear objetos ---
mi_tanque = Tanque(A=2.5, a=1)
sim = SimulacionNivel(modelo_tanque=mi_tanque, dt=0.01, h0=0.0)

# --- 3. Resultados simulacion ---
time, niveles = sim.ejecutar(q_in_signal)
h = niveles
#----modelo de lnn-----
model = LiquidTank(1, 20)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

h_state = torch.zeros(1, 20)

epochs = 50

for epoch in range(epochs):

    total_loss = 0
    h_state = torch.zeros(1, 20)

    for i in range(len(time) - 1):
        q = torch.tensor([[q_in_signal[i]]], dtype=torch.float32)
        target = torch.tensor([[h[i + 1]]], dtype=torch.float32)

        # IMPORTANTE: cortar el grafo
        h_state = h_state.detach()

        pred, h_state = model(q, h_state)

        loss = criterion(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch}, Loss: {total_loss}")

h_state = torch.zeros(1, 20)
predictions = []

for i in range(len(time)-1):
    q = torch.tensor([[q_in_signal[i]]], dtype=torch.float32)
    pred, h_state = model(q, h_state)
    predictions.append(pred.item())

plt.plot(time[1:], h[1:], label="Real")
plt.plot(time[1:], predictions, label="LNN")
plt.legend()
plt.title("Comparación Real vs LNN")
plt.show()