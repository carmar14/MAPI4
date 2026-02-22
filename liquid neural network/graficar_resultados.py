import json
import matplotlib.pyplot as plt
import torch

with open("resultados_lnn_.json", "r") as f:
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

# Graficar
plt.figure(figsize=(10,6))

# Real
plt.plot(time_train, real_train, label="Real Train", linestyle="-")
plt.plot(time_val, real_val, label="Real Val", linestyle="--")

# Pred
plt.plot(time_train, pred_train, label="Pred Train", linestyle="-")
plt.plot(time_val, pred_val, label="Pred Val", linestyle="--")

# Línea vertical que marca el split
plt.axvline(x=time[split_idx], color='gray', linestyle=':', label="Train/Val Split")

plt.legend()
plt.xlabel("Tiempo")
plt.ylabel("Nivel h")
plt.title("Predicción vs Real (Train / Validation)")
plt.grid(True)
plt.show()
'''
plt.figure()
plt.plot(time[1:], h[1:], label="Real")
plt.plot(time, predictions, label="LNN")
plt.legend()
plt.title("Comparación Real vs LNN")
plt.show()
'''
plt.show()