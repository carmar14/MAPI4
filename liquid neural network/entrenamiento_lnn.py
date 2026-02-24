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
noise_std = 0.02 * np.std(niveles)   # 2% del desvío estándar de la señal
h = niveles #+ np.random.normal(0, noise_std, size=niveles.shape)
#----modelo de lnn-----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LiquidTank(2, 20)
optimizer = optim.Adam(model.parameters(), lr=0.001)#, weight_decay=1e-4)
criterion = nn.MSELoss() #nn.HuberLoss(delta=1.0)

h_state = torch.zeros(1, 20)

epochs = 50
#loss_history = []
train_loss_history = []
val_loss_history = []

#----preparar training and validation
N = len(time)
train_ratio = 0.7
train_size = int(N * train_ratio)

# Train
time_train = time[:train_size]
h_train = h[:train_size]
q_train = q_in_signal[:train_size]

# Validation
time_val = time[train_size:]
h_val = h[train_size:]
q_val = q_in_signal[train_size:]

#Normalizar entradas y salidas
q_mean, q_std = np.mean(q_train), np.std(q_train)
h_mean, h_std = np.mean(h_train), np.std(h_train)

q_train_norm = (q_train - q_mean) / q_std
h_train_norm = (h_train - h_mean) / h_std

q_val_norm = (q_val - q_mean) / q_std
h_val_norm = (h_val - h_mean) / h_std

patience = 5
best_val_loss = float("inf")
counter = 0

for epoch in range(epochs):
    model.train()
    total_loss = 0
    h_state = torch.zeros(1, 20)

    for i in range(len(time_train) - 1):
        optimizer.zero_grad()
        '''
        q = torch.tensor([[q_in_signal[i]]], dtype=torch.float32)
        target = torch.tensor([[h[i + 1]]], dtype=torch.float32)
        '''
        #q = torch.tensor([[q_train[i]]], dtype=torch.float32)
        q = torch.tensor([[q_train[i], h_train[i]]], dtype=torch.float32)
        target = torch.tensor([[h_train[i + 1]]], dtype=torch.float32)

        # IMPORTANTE: cortar el grafo
        h_state = h_state.detach()

        pred, h_state = model(q, h_state)

        loss = criterion(pred, target)

        #optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
    train_loss_history.append(total_loss)
    #validar
    model.eval()
    val_loss = 0
    h_state_val = torch.zeros(1, 20)
    h_current = torch.tensor([[h_val[0]]], dtype=torch.float32)
    with torch.no_grad():
        for i in range(len(time_val) - 1):
            #q = torch.tensor([[q_val[i]]], dtype=torch.float32)
            q = torch.tensor([[q_val[i], h_val[i]]], dtype=torch.float32)
            target = torch.tensor([[h_val[i + 1]]], dtype=torch.float32)
            h_state_val = h_state_val.detach()
            pred, h_state_val = model(q, h_state_val)

            loss = criterion(pred, target)
            val_loss += loss.item()
    val_loss_history.append(val_loss)

    print(f"Epoch {epoch} | Train Loss: {total_loss:.4f} | Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        # Guardar mejor modelo
        torch.save(model.state_dict(), "best_lnn_model.pth")

    else:
        counter += 1
        print(f"No improvement for {counter} epochs")

    if counter >= patience:
        print("Early stopping activated")
        break


    #print(f"Epoch {epoch}, Loss: {total_loss}")

h_state = torch.zeros(1, 20)
predictions = []

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
    "h_pred": predictions,
    "train_loss": train_loss_history,
    "val_loss": val_loss_history
}

# Guardar archivo
with open("resultados_lnn.json", "w") as f:
    json.dump(results, f, indent=4)

print("Resultados guardados en resultados_lnn.json")

torch.save(model.state_dict(), "liquid_tank_model.pth")

'''
plt.plot(time[1:], h[1:], label="Real")
plt.plot(time[1:], predictions, label="LNN")
plt.legend()
plt.title("Comparación Real vs LNN")
plt.show()
'''