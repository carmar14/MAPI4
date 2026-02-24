import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

with open("resultados_lnn_ruido.json", "r") as f:
    data = json.load(f)

#train_loss = data["train_loss"]
#val_loss = data["val_loss"]
time = data["time"]
h = data["h_real"]
predictions = data["h_pred"]
print(len(predictions))
print(len(time))
'''
epochs = range(len(train_loss))

plt.plot(train_loss, label="Train Loss")
plt.plot(val_loss, label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("Curva de entrenamiento LNN")
plt.legend()
plt.grid(True)
'''

N = len(time)
train_ratio = 0.7
train_size = int(N * train_ratio)
split_idx = train_size

# Separar
time_train = time[:split_idx]
time_val = time[split_idx:]

real_train = h[:split_idx]
real_val = h[split_idx:]

pred_train = predictions[:split_idx]
pred_val = predictions[split_idx:]

mse = torch.mean((torch.tensor(pred_val)- torch.tensor(real_val)) ** 2)*100
print("MSE:", mse.item())

# Configuración de la figura principal
fig, ax = plt.subplots(figsize=(12, 7))

# Graficación de datos reales
ax.plot(time_train, real_train, label="Ground Truth (Training)", color='blue', linewidth=1.5)
ax.plot(time_val, real_val, label="Ground Truth (Validation)", color= 'green', linestyle='--', linewidth=1.5)

# Graficación de predicciones LNN
ax.plot(time_train, pred_train, label="LNN Prediction (Training)", color='red', linewidth=1)
ax.plot(time_val, pred_val, label="LNN Prediction (Validation)", color= 'orange', linestyle='--', linewidth=1)

# Línea de división entre entrenamiento y validación
ax.axvline(x=time[split_idx], color='gray', linestyle=':', label="Training and Validation Boundary")

# Etiquetas académicas (English - No contractions)
ax.set_xlabel("Time [s]", fontsize=12)
ax.set_ylabel("Liquid Level [m]", fontsize=12)
ax.set_title("Recursive State Estimation with sensor noise: Prediction versus Ground Truth", fontsize=14)
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, linestyle='--', alpha=0.7)

# --- ZOOM 1: Zona de Entrenamiento ---
# Se crea el eje insertado en la zona inferior izquierda (loc='lower left')
axins1 = inset_axes(ax, width="15%", height="15%", loc='lower left',
                   bbox_to_anchor=(0.1, 0.1, 1, 1), bbox_transform=ax.transAxes)
axins1.plot(time_train, real_train, color='blue', alpha=0.6)
axins1.plot(time_train, pred_train, color='red')

# Definir los límites del zoom (ajustar según tus datos)
x1, x2, y1, y2 = 15, 25, 0.04, 0.1
axins1.set_xlim(x1, x2)
axins1.set_ylim(y1, y2)
axins1.set_xticklabels([]) # Quitar etiquetas para limpieza visual
axins1.set_yticklabels([])
mark_inset(ax, axins1, loc1=2, loc2=4, fc="none", ec="0.5") # Conecta el zoom con el área real

# --- ZOOM 2: Zona de Validación ---
# Se crea el eje insertado en la zona inferior derecha (loc='lower right')
axins2 = inset_axes(ax, width="15%", height="15%", loc='lower right',
                   bbox_to_anchor=(-0.1, 0.1, 1, 1), bbox_transform=ax.transAxes)
axins2.plot(time_val, real_val, color= 'green', linestyle='--')
axins2.plot(time_val, pred_val, color= 'orange', linestyle='--')

# Definir los límites del zoom (ajustar según tus datos)
x3, x4, y3, y4 = 285, 295, 0.085, 0.11
axins2.set_xlim(x3, x4)
axins2.set_ylim(y3, y4)
axins2.set_xticklabels([])
axins2.set_yticklabels([])
mark_inset(ax, axins2, loc1=1, loc2=3, fc="none", ec="0.5")

'''
# Graficar
plt.figure(figsize=(10,6))

# Real
plt.plot(time_train, real_train, label="Ground Truth Train", linestyle="-")
plt.plot(time_val, real_val, label="Ground Truth Val", linestyle="--")

# Pred
plt.plot(time_train, pred_train, label="Pred Train", linestyle="-")
plt.plot(time_val, pred_val, label="Pred Val", linestyle="--")

# Línea vertical que marca el split
plt.axvline(x=time[split_idx], color='gray', linestyle=':', label="Train/Val Split")

plt.legend()
plt.xlabel("Time(seconds)")
plt.ylabel("Liquid Level (m)")
plt.title("Prediction vs Ground Truth (Train / Validation)")
plt.grid(True)
plt.show()
'''

'''
plt.figure()
plt.plot(time[1:], h[1:], label="Real")
plt.plot(time, predictions, label="LNN")
plt.legend()
plt.title("Comparación Real vs LNN")
plt.show()
'''
plt.show()