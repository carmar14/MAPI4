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

# --- modelo sencillo LSTM para comparación ---
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

# --- modelo sencillo RNN para comparación ---
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
def generar_escalones_random(n_segmentos=50, duracion=300):
    valores = np.random.uniform(0.5, 2.0, n_segmentos)
    return np.repeat(valores, duracion)

# --- menú interactivo para seleccionar modelos ---
print("\nModel selection:")
print(" 1) Train LNN")
print(" 2) Train simple LSTM")
print(" 3) Train simple RNN")
print("Enter numbers separated by comma (e.g. 1,3 to train LNN and RNN)")
choice = input("Your choice: ")
selected = set(ch.strip() for ch in choice.split(",") if ch.strip())
train_lnn = '1' in selected
train_lstm = '2' in selected
train_rnn = '3' in selected
if not (train_lnn or train_lstm or train_rnn):
    print("No valid model selected; exiting.")
    exit(0)


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
train_loss_history = []
val_loss_history = []
if train_lnn:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LiquidTank(2, 20)
    optimizer = optim.Adam(model.parameters(), lr=0.001)#, weight_decay=1e-4)
    criterion = nn.MSELoss() #nn.HuberLoss(delta=1.0)
    h_state = torch.zeros(1, 20)
    epochs = 50
else:
    print("Skipping LNN training")

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

if train_lnn:
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        h_state = torch.zeros(1, 20)

        for i in range(len(time_train) - 1):
            optimizer.zero_grad()
            q = torch.tensor([[q_train[i], h_train[i]]], dtype=torch.float32)
            target = torch.tensor([[h_train[i + 1]]], dtype=torch.float32)

            # IMPORTANTE: cortar el grafo
            h_state = h_state.detach()

            pred, h_state = model(q, h_state)

            loss = criterion(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
        train_loss_history.append(total_loss)
        # validar
        model.eval()
        val_loss = 0
        h_state_val = torch.zeros(1, 20)
        with torch.no_grad():
            for i in range(len(time_val) - 1):
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
            torch.save(model.state_dict(), "best_lnn_model.pth")
        else:
            counter += 1
            print(f"No improvement for {counter} epochs")
        if counter >= patience:
            print("Early stopping activated")
            break

    h_state = torch.zeros(1, 20)
    predictions = []
    model.load_state_dict(torch.load("best_lnn_model.pth"))
    model.eval()
    for i in range(len(time)-1):
        q = torch.tensor([[q_in_signal[i], h[i]]], dtype=torch.float32)
        pred, h_state = model(q, h_state)
        predictions.append(pred.item())
    results = {
        "time": time[1:].tolist(),
        "h_real": h[1:].tolist(),
        "h_pred": predictions,
        "train_loss": train_loss_history,
        "val_loss": val_loss_history
    }
    with open("resultados_lnn.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Resultados guardados en resultados_lnn.json")
    torch.save(model.state_dict(), "liquid_tank_model.pth")
else:
    results = None
    print("LNN training skipped")

'''
plt.plot(time[1:], h[1:], label="Real")
plt.plot(time[1:], predictions, label="LNN")
plt.legend()
plt.title("Comparación Real vs LNN")
plt.show()
'''

# --- 4. Entrenamiento y evaluación de la LSTM simple ---
if train_lstm:
    print("\n--- Iniciando entrenamiento de LSTM para comparación ---")
    lstm_model = SimpleLSTM(2, 20)
    lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
    lstm_criterion = nn.MSELoss()
    lstm_train_loss = []
    lstm_val_loss = []
    best_val_loss_lstm = float("inf")
    counter_lstm = 0
    patience_lstm = 5
    epochs_lstm = 50
    for epoch in range(epochs_lstm):
        lstm_model.train()
        total_loss = 0
        lstm_h = None
        for i in range(len(time_train) - 1):
            lstm_optimizer.zero_grad()
            q = torch.tensor([[q_train[i], h_train[i]]], dtype=torch.float32).unsqueeze(0)
            target = torch.tensor([[h_train[i + 1]]], dtype=torch.float32)
            if lstm_h is not None:
                lstm_h = (lstm_h[0].detach(), lstm_h[1].detach())
            pred, lstm_h = lstm_model(q, lstm_h)
            loss = lstm_criterion(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lstm_model.parameters(), 1.0)
            lstm_optimizer.step()
            total_loss += loss.item()
        lstm_train_loss.append(total_loss)
        # validación
        lstm_model.eval()
        val_loss = 0
        lstm_h_val = None
        with torch.no_grad():
            for i in range(len(time_val) - 1):
                q = torch.tensor([[q_val[i], h_val[i]]], dtype=torch.float32).unsqueeze(0)
                target = torch.tensor([[h_val[i + 1]]], dtype=torch.float32)
                if lstm_h_val is not None:
                    lstm_h_val = (lstm_h_val[0].detach(), lstm_h_val[1].detach())
                pred, lstm_h_val = lstm_model(q, lstm_h_val)
                val_loss += lstm_criterion(pred, target).item()
        lstm_val_loss.append(val_loss)
        print(f"LSTM Epoch {epoch} | Train Loss: {total_loss:.4f} | Val Loss: {val_loss:.4f}")
        if val_loss < best_val_loss_lstm:
            best_val_loss_lstm = val_loss
            counter_lstm = 0
            torch.save(lstm_model.state_dict(), "best_lstm_model.pth")
        else:
            counter_lstm += 1
            print(f"LSTM no improvement for {counter_lstm} epochs")
        if counter_lstm >= patience_lstm:
            print("Early stopping LSTM activated")
            break
    # predicciones LSTM
    lstm_h = None
    lstm_preds = []
    lstm_model.load_state_dict(torch.load("best_lstm_model.pth"))
    lstm_model.eval()
    for i in range(len(time) - 1):
        q = torch.tensor([[q_in_signal[i], h[i]]], dtype=torch.float32).unsqueeze(0)
        pred, lstm_h = lstm_model(q, lstm_h)
        lstm_preds.append(pred.item())
    lstm_results = {
        "time": time[1:].tolist(),
        "h_real": h[1:].tolist(),
        "h_pred": lstm_preds,
        "train_loss": lstm_train_loss,
        "val_loss": lstm_val_loss
    }
    with open("resultados_lstm.json", "w") as f:
        json.dump(lstm_results, f, indent=4)
    print("Resultados guardados en resultados_lstm.json")
else:
    print("LSTM training skipped")

# comparación de MSE en validación
if train_lnn and results is not None:
    mse_lnn = torch.mean((torch.tensor(results["h_pred"][train_size:]) - torch.tensor(results["h_real"][train_size:]))**2)
    print(f"Validation MSE LNN: {mse_lnn.item():.6f}")
else:
    mse_lnn = None
    print("LNN metrics skipped")

if train_lstm:
    mse_lstm = torch.mean((torch.tensor(lstm_results["h_pred"][train_size:]) - torch.tensor(lstm_results["h_real"][train_size:]))**2)
    print(f"Validation MSE LSTM: {mse_lstm.item():.6f}")
else:
    mse_lstm = None
    print("LSTM metrics skipped")

# --- 5. Entrenamiento y evaluación de la RNN simple ---
if train_rnn:
    print("\n--- Iniciando entrenamiento de RNN para comparación ---")
    rnn_model = SimpleRNN(2, 20)
    rnn_optimizer = optim.Adam(rnn_model.parameters(), lr=0.001)
    rnn_criterion = nn.MSELoss()
    rnn_train_loss = []
    rnn_val_loss = []
    best_val_loss_rnn = float("inf")
    counter_rnn = 0
    patience_rnn = 5
    epochs_rnn = 50
    for epoch in range(epochs_rnn):
        rnn_model.train()
        total_loss = 0
        rnn_h = None
        for i in range(len(time_train) - 1):
            rnn_optimizer.zero_grad()
            q = torch.tensor([[q_train[i], h_train[i]]], dtype=torch.float32).unsqueeze(0)
            target = torch.tensor([[h_train[i + 1]]], dtype=torch.float32)
            if rnn_h is not None:
                rnn_h = rnn_h.detach()
            pred, rnn_h = rnn_model(q, rnn_h)
            loss = rnn_criterion(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rnn_model.parameters(), 1.0)
            rnn_optimizer.step()
            total_loss += loss.item()
        rnn_train_loss.append(total_loss)
        # validación
        rnn_model.eval()
        val_loss = 0
        rnn_h_val = None
        with torch.no_grad():
            for i in range(len(time_val) - 1):
                q = torch.tensor([[q_val[i], h_val[i]]], dtype=torch.float32).unsqueeze(0)
                target = torch.tensor([[h_val[i + 1]]], dtype=torch.float32)
                if rnn_h_val is not None:
                    rnn_h_val = rnn_h_val.detach()
                pred, rnn_h_val = rnn_model(q, rnn_h_val)
                val_loss += rnn_criterion(pred, target).item()
        rnn_val_loss.append(val_loss)
        print(f"RNN Epoch {epoch} | Train Loss: {total_loss:.4f} | Val Loss: {val_loss:.4f}")
        if val_loss < best_val_loss_rnn:
            best_val_loss_rnn = val_loss
            counter_rnn = 0
            torch.save(rnn_model.state_dict(), "best_rnn_model.pth")
        else:
            counter_rnn += 1
            print(f"RNN no improvement for {counter_rnn} epochs")
        if counter_rnn >= patience_rnn:
            print("Early stopping RNN activated")
            break
    # predicciones RNN
    rnn_h = None
    rnn_preds = []
    rnn_model.load_state_dict(torch.load("best_rnn_model.pth"))
    rnn_model.eval()
    for i in range(len(time) - 1):
        q = torch.tensor([[q_in_signal[i], h[i]]], dtype=torch.float32).unsqueeze(0)
        pred, rnn_h = rnn_model(q, rnn_h)
        rnn_preds.append(pred.item())
    rnn_results = {
        "time": time[1:].tolist(),
        "h_real": h[1:].tolist(),
        "h_pred": rnn_preds,
        "train_loss": rnn_train_loss,
        "val_loss": rnn_val_loss
    }
    with open("resultados_rnn.json", "w") as f:
        json.dump(rnn_results, f, indent=4)
    print("Resultados guardados en resultados_rnn.json")
    mse_rnn = torch.mean((torch.tensor(rnn_results["h_pred"][train_size:]) - torch.tensor(rnn_results["h_real"][train_size:]))**2)
    print(f"Validation MSE RNN: {mse_rnn.item():.6f}")
else:
    print("RNN training skipped")
