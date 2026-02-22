import json
import matplotlib.pyplot as plt

with open("resultados_lnn.json", "r") as f:
    data = json.load(f)

train_loss = data["train_loss"]
val_loss = data["val_loss"]
time = data["time"]
h = data["h_real"]
predictions = data["h_pred"]
print(len(predictions))
print(len(time))

epochs = range(len(train_loss))

plt.plot(train_loss, label="Train Loss")
plt.plot(val_loss, label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("Curva de entrenamiento LNN")
plt.legend()
plt.grid(True)

plt.figure()
plt.plot(time[1:], h[1:], label="Real")
plt.plot(time, predictions, label="LNN")
plt.legend()
plt.title("Comparación Real vs LNN")
plt.show()
plt.show()