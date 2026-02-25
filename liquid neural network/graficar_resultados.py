import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# elegir variante: con ruido (_ruido) o sin ruido
print("¿Qué resultados desea graficar?")
print(" 1) Con ruido (archivos *_ruido.json)")
print(" 2) Sin ruido (archivos .json sin sufijo)")
choice = input("Elija 1 o 2 (por defecto 1): ").strip() or "1"
if choice == "2":
    suffix = ""
else:
    suffix = "_ruido"

# intentar cargar archivo combinado primero
combined_path = f"resultados_modelo{suffix}.json"
if os.path.isfile(combined_path):
    with open(combined_path, "r") as f:
        data = json.load(f)
    positive_path_info = f"(loaded {combined_path})"
    print("Using combined results", positive_path_info)
else:
    # fallback: cargar archivos por modelo (LNN)
    lnn_candidates = [f"resultados_lnn{suffix}.json", f"liquid neural network/resultados_lnn{suffix}.json", f"resultados_lnn.json"]
    lnn_path = None
    for p in lnn_candidates:
        if os.path.isfile(p):
            lnn_path = p
            break
    if lnn_path is None:
        raise FileNotFoundError("No se encontró archivo de resultados LNN")
    with open(lnn_path, "r") as f:
        data = json.load(f)
    positive_path_info = f"(loaded {lnn_path})"
    print("Using LNN results", positive_path_info)

# load LSTM results if available
lstm_path = f"resultados_lstm{suffix}.json"
if os.path.isfile(lstm_path):
    with open(lstm_path, "r") as f:
        lstm_data = json.load(f)
else:
    lstm_data = None

# load RNN results if available
rnn_path = f"resultados_rnn{suffix}.json"
if os.path.isfile(rnn_path):
    with open(rnn_path, "r") as f:
        rnn_data = json.load(f)
else:
    rnn_data = None

#train_loss = data["train_loss"]
#val_loss = data["val_loss"]
time = data["time"]
h = data["h_real"]
predictions = data["h_pred"]
if lstm_data is not None:
    lstm_predictions = lstm_data.get("h_pred", [])
else:
    lstm_predictions = []
if rnn_data is not None:
    rnn_predictions = rnn_data.get("h_pred", [])
else:
    rnn_predictions = []

# --- menú para elegir qué modelos dibujar ---
print("\nModelos disponibles:")
print(f" LNN: {'sí' if data is not None else 'no'}")
print(f" LSTM: {'sí' if lstm_data is not None else 'no'}")
print(f" RNN: {'sí' if rnn_data is not None else 'no'}")
print("Elija números separados por coma (ej. 1,3 para LNN y RNN):")
print(" 1) LNN")
print(" 2) LSTM")
print(" 3) RNN")
choice = input("Su elección: ")
selected = set(ch.strip() for ch in choice.split(",") if ch.strip())
show_lnn = '1' in selected and data is not None
show_lstm = '2' in selected and lstm_data is not None
show_rnn = '3' in selected and rnn_data is not None
if '1' in selected and not show_lnn:
    print("LNN no disponible, se omitirá.")
if '2' in selected and not show_lstm:
    print("LSTM no disponible, se omitirá.")
if '3' in selected and not show_rnn:
    print("RNN no disponible, se omitirá.")
if not (show_lnn or show_lstm or show_rnn):
    print("No hay modelos seleccionados. Saliendo.")
    exit(0)

print(f"Modelos que se graficarán: LNN={show_lnn}, LSTM={show_lstm}, RNN={show_rnn}")

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

# compute MSEs using consistent validation set
if show_lnn:
    mse = torch.mean((torch.tensor(pred_val)- torch.tensor(real_val)) ** 2)*100
    print("LNN MSE:", mse.item())
    # compute R^2 for LNN
    ss_res = torch.sum((torch.tensor(real_val) - torch.tensor(pred_val))**2)
    ss_tot = torch.sum((torch.tensor(real_val) - torch.mean(torch.tensor(real_val)))**2)
    r2_lnn = 1 - ss_res/ss_tot if ss_tot>0 else float('nan')
    print("LNN R2:", r2_lnn.item() if isinstance(r2_lnn, torch.Tensor) else r2_lnn)
else:
    ss_tot = torch.sum((torch.tensor(real_val) - torch.mean(torch.tensor(real_val)))**2)
    print("Saltando métricas LNN")

if show_lstm and len(lstm_predictions)==len(time):
    lstm_pred_val = lstm_predictions[split_idx:]
    mse_lstm = torch.mean((torch.tensor(lstm_pred_val)- torch.tensor(real_val)) ** 2)*100
    print("LSTM MSE:", mse_lstm.item())
    ss_res_l = torch.sum((torch.tensor(real_val) - torch.tensor(lstm_pred_val))**2)
    r2_lstm = 1 - ss_res_l/ss_tot if ss_tot>0 else float('nan')
    print("LSTM R2:", r2_lstm.item() if isinstance(r2_lstm, torch.Tensor) else r2_lstm)
else:
    print("Saltando métricas LSTM")

if show_rnn and len(rnn_predictions)==len(time):
    rnn_pred_val = rnn_predictions[split_idx:]
    mse_rnn = torch.mean((torch.tensor(rnn_pred_val)- torch.tensor(real_val)) ** 2)*100
    print("RNN MSE:", mse_rnn.item())
    ss_res_r = torch.sum((torch.tensor(real_val) - torch.tensor(rnn_pred_val))**2)
    r2_rnn = 1 - ss_res_r/ss_tot if ss_tot>0 else float('nan')
    print("RNN R2:", r2_rnn.item() if isinstance(r2_rnn, torch.Tensor) else r2_rnn)
else:
    print("Saltando métricas RNN")

# Configuración de la figura principal
fig, ax = plt.subplots(figsize=(12, 7))

# Graficación de datos reales
ax.plot(time_train, real_train, label="Ground Truth (Training)", color='blue', linewidth=1.5)
ax.plot(time_val, real_val, label="Ground Truth (Validation)", color= 'green', linestyle='--', linewidth=1.5)

# Graficación de predicciones LNN
if show_lnn:
    ax.plot(time_train, pred_train, label="LNN Prediction (Training)", color='red', linewidth=1)
    ax.plot(time_val, pred_val, label="LNN Prediction (Validation)", color= 'orange', linestyle='--', linewidth=1)

# Graficación de predicciones LSTM si están seleccionadas
if show_lstm and lstm_data is not None and len(lstm_predictions)==len(time):
    lstm_train = lstm_predictions[:split_idx]
    lstm_val = lstm_predictions[split_idx:]
    ax.plot(time_train, lstm_train, label="LSTM Prediction (Training)", color='purple', linewidth=1)
    ax.plot(time_val, lstm_val, label="LSTM Prediction (Validation)", color='magenta', linestyle='--', linewidth=1)
# Graficación de predicciones RNN si están seleccionadas
if show_rnn and rnn_data is not None and len(rnn_predictions)==len(time):
    rnn_train = rnn_predictions[:split_idx]
    rnn_val = rnn_predictions[split_idx:]
    ax.plot(time_train, rnn_train, label="RNN Prediction (Training)", color='brown', linewidth=1)
    ax.plot(time_val, rnn_val, label="RNN Prediction (Validation)", color='orange', linestyle='-.', linewidth=1)

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
if show_lstm and lstm_data is not None and len(lstm_predictions)>=split_idx:
    axins1.plot(time_train, lstm_train, color='purple', alpha=0.6)
if show_rnn and rnn_data is not None and len(rnn_predictions)>=split_idx:
    axins1.plot(time_train, rnn_train, color='brown', alpha=0.6)

# Definir los límites del zoom (ajustar según tus datos)
x1, x2, y1, y2 = 15, 25, 0.04, 0.1
axins1.set_xlim(x1, x2)
axins1.set_ylim(y1, y2)
axins1.set_xticklabels([]) # Quitar etiquetas para limpieza visual
axins1.set_yticklabels([])
mark_inset(ax, axins1, loc1=2, loc2=4, fc="none", ec="0.5") # Conecta el zoom con el área real

# --- ZOOM 2: Zona de Validación (últimos instantes) ---
if show_lnn or show_lstm or show_rnn:
    axins2 = inset_axes(ax, width="15%", height="15%", loc='lower right',
                       bbox_to_anchor=(-0.1, 0.1, 1, 1), bbox_transform=ax.transAxes)
    axins2.plot(time_val, real_val, color= 'green', linestyle='--')
    axins2.plot(time_val, pred_val, color= 'orange', linestyle='--')
    if show_lstm and lstm_data is not None and len(lstm_predictions)>=len(time_val):
        axins2.plot(time_val, lstm_val, color='magenta', linestyle='--')
    if show_rnn and rnn_data is not None and len(rnn_predictions)>=len(time_val):
        axins2.plot(time_val, rnn_val, color='brown', linestyle='-.')
    # límites dinámicos: últimos 10% de validación
    Nval = len(time_val)
    if Nval>5:
        j1 = int(Nval*0.9)
        j2 = Nval-1
        x3, x4 = time_val[j1], time_val[j2]
    else:
        x3, x4 = time_val[0], time_val[-1]
    vals = real_val[j1:j2] + pred_val[j1:j2]
    if show_lstm and lstm_data is not None:
        vals += lstm_val[j1:j2]
    if show_rnn and rnn_data is not None:
        vals += rnn_val[j1:j2]
    y3, y4 = min(vals), max(vals)
    dy = (y4-y3)*0.1 if y4>y3 else 0.01
    axins2.set_xlim(x3, x4)
    axins2.set_ylim(y3-dy, y4+dy)
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