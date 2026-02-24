from lnn import LiquidTank
import torch
from componentes import Tanque
from simulacion import SimulacionNivel
import numpy as np
import json
import matplotlib.pyplot as plt

# modelos auxiliares de comparación (definidos aquí para no ejecutar el entrenamiento)
import torch.nn as nn

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, h=None):
        # x: (seq_len, batch, input_size)
        out, h_out = self.lstm(x, h)
        y = self.fc(out[-1])      # usamos la última salida
        return y, h_out

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, nonlinearity='tanh')
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, h=None):
        # x: (seq_len, batch, input_size)
        out, h_out = self.rnn(x, h)
        y = self.fc(out[-1])
        return y, h_out

# --- 1. Configuración de entrada (Escalones) ---
def generar_escalones_random(n_segmentos=20, duracion=300):
    valores = np.random.uniform(0.5, 2.0, n_segmentos)
    return np.repeat(valores, duracion)

q_in_signal = generar_escalones_random(20, 400)
#q_in_signal = np.repeat([0.8, 1.2, 0.8, 2.0, 1.5], 700) # Amplitudes variables

# --- 2. Crear objetos ---
mi_tanque = Tanque(A=2.8, a=1)
sim = SimulacionNivel(modelo_tanque=mi_tanque, dt=0.01, h0=0.0)

# --- 3. Resultados simulacion ---
time, niveles = sim.ejecutar(q_in_signal)
noise_std = 0.02 * np.std(niveles)   # 2% del desvío estándar de la señal
h = niveles + np.random.normal(0, noise_std, size=niveles.shape)
h_state = torch.zeros(1, 20)


#plt.plot(time, h, label="Nivel con ruido")
#plt.show()

# prepare containers for each model's predictions
predictions = {
    "lnn": [],
    "lstm": [],
    "rnn": []
}

# load LNN model if available
try:
    model_lnn = LiquidTank(2, 20)
    model_lnn.load_state_dict(torch.load("best_lnn_model.pth"))
    model_lnn.eval()
    has_lnn = True
except Exception as e:
    print("Advertencia: no se pudo cargar modelo LNN:", e)
    has_lnn = False


try:
    model_lstm = SimpleLSTM(2, 20)
    model_lstm.load_state_dict(torch.load("best_lstm_model.pth"))
    model_lstm.eval()
    has_lstm = True
except Exception as e:
    print("Advertencia: no se pudo cargar modelo LSTM:", e)
    has_lstm = False

try:
    model_rnn = SimpleRNN(2, 20)
    model_rnn.load_state_dict(torch.load("best_rnn_model.pth"))
    model_rnn.eval()
    has_rnn = True
except Exception as e:
    print("Advertencia: no se pudo cargar modelo RNN:", e)
    has_rnn = False

# initialize hidden states
h_state_lnn = torch.zeros(1, 20)
h_state_lstm = None
h_state_rnn = None

# run inference loop
for i in range(len(time) - 1):
    q_val = q_in_signal[i]
    h_val = h[i]

    if has_lnn:
        q_tensor = torch.tensor([[q_val, h_val]], dtype=torch.float32)
        pred, h_state_lnn = model_lnn(q_tensor, h_state_lnn)
        predictions["lnn"].append(pred.item())

    if has_lstm:
        q_tensor = torch.tensor([[q_val, h_val]], dtype=torch.float32).unsqueeze(0)
        pred, h_state_lstm = model_lstm(q_tensor, h_state_lstm)
        predictions["lstm"].append(pred.item())

    if has_rnn:
        q_tensor = torch.tensor([[q_val, h_val]], dtype=torch.float32).unsqueeze(0)
        pred, h_state_rnn = model_rnn(q_tensor, h_state_rnn)
        predictions["rnn"].append(pred.item())

# build results dict
results = {
    "time": time[1:].tolist(),
    "h_real": h[1:].tolist()
}
if has_lnn:
    results["h_pred_lnn"] = predictions["lnn"]
if has_lstm:
    results["h_pred_lstm"] = predictions["lstm"]
if has_rnn:
    results["h_pred_rnn"] = predictions["rnn"]
# save per-model files (incluye la señal real y el tiempo)
if has_lnn:
    results_lnn = {
        "time": time[1:].tolist(),
        "h_real": h[1:].tolist(),
        "h_pred": predictions["lnn"]
    }
    with open("resultados_lnn_ruido.json", "w") as f:
        json.dump(results_lnn, f, indent=4)

if has_lstm:
    results_lstm = {
        "time": time[1:].tolist(),
        "h_real": h[1:].tolist(),
        "h_pred": predictions["lstm"]
    }
    with open("resultados_lstm_ruido.json", "w") as f:
        json.dump(results_lstm, f, indent=4)

if has_rnn:
    results_rnn = {
        "time": time[1:].tolist(),
        "h_real": h[1:].tolist(),
        "h_pred": predictions["rnn"]
    }
    with open("resultados_rnn_ruido.json", "w") as f:
        json.dump(results_rnn, f, indent=4)

# lista de archivos guardados (solo por modelo)
saved = []
if has_lnn:
    saved.append("resultados_lnn_ruido.json")
if has_lstm:
    saved.append("resultados_lstm_ruido.json")
if has_rnn:
    saved.append("resultados_rnn_ruido.json")

print("Resultados guardados:", ", ".join(saved))