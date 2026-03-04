import matplotlib.pyplot as plt
import numpy as np
from componentes import Tanque
from simulacion import SimulacionNivel
import serial
import time
from sklearn.metrics import mean_squared_error

# Configura el puerto según tu sistema (COMx en Windows, /dev/ttyUSBx en Linux)
ser = serial.Serial('COM3', 115200, timeout=1)
time.sleep(2) # Esperar a que Arduino se reinicie

'''
# 2. Configuración de la Gráfica
plt.ion() # Activar modo interactivo
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title("Inferencia LNN en Vivo: Arduino vs Objetivo")
ax.set_xlabel("Tiempo (muestras)")
ax.set_ylabel("Amplitud")

# Creamos dos líneas vacías
line_target, = ax.plot([], [], 'r-', label="Objetivo (Python)", alpha=0.6)
line_pred, = ax.plot([], [], 'b--', label="LNN (Arduino)", lw=2)
ax.legend(loc="upper right")
ax.grid(True)

# Listas para almacenar los datos que se van acumulando
datos_target = []
datos_pred = []
ventana_visualizacion = 100 # Cuántas muestras mostrar a la vez
'''

# --- 1. Configuración de entrada (Escalones) ---
#q_in = np.repeat([0.8, 1.2],700)#
q_in = np.repeat([0.8, 1.2, 0.75, 1.18, 1.3, 0.9], 700) # Amplitudes variables, valores por encima de 1.5 se hace mas evidente el error

# --- 2. Crear objetos ---
mi_tanque = Tanque(A=2.5, a=1)
sim = SimulacionNivel(modelo_tanque=mi_tanque, dt=0.01, h0=0.0)

# --- 3. Correr y Graficar ---
tiempo, niveles = sim.ejecutar(q_in)
noise_std = 0.02 * np.std(niveles)   # 2% del desvío estándar de la señal
niveles = niveles + np.random.normal(0, noise_std, size=niveles.shape)  # 2% del desvío estándar de la señal
#plt.plot(tiempo, niveles)
#plt.show()
u0_sim = q_in[:-1]
u1_sim = niveles[:-1]
y_expected = niveles[1:]# Simulación de la salida ideal
y_pred_data = []
y_true_data = []

print(u0_sim.size, u1_sim.size, y_expected.size)

try:
    for i in range(u0_sim.size):
        # Enviar datos al Arduino: "u0,u1,y_true\n"

        payload = f"{u0_sim[i]:.4f},{u1_sim[i]:.4f},{y_expected[i]:.4f}\n"
        ser.write(payload.encode())
        print(i, payload)
        # Leer respuesta
        line = ser.readline().decode('utf-8').strip()
        print(line)
        if line:
            parts = line.split(',')
            if len(parts) == 2:
                y_t = float(parts[0])
                y_p = float(parts[1])

                y_true_data.append(y_t)
                y_pred_data.append(y_p)



except KeyboardInterrupt:
    ser.close()
    print("Simulación terminada.")


plt.figure(figsize=(12, 6))
plt.plot(y_true_data, label='Target (Python)', color='black', lw=1.5)
plt.plot(y_pred_data, label='Inferencia LNN (Arduino)', color='red', linestyle='--', lw=1.5)

plt.title('Comparación: Salida Esperada vs Predicción LNN en Hardware')
plt.xlabel('Muestras (Time Steps)')
plt.ylabel('Valor de Salida')
plt.legend()
plt.grid(True, alpha=0.3)

# Calcular error
mse = mean_squared_error(y_true_data, y_pred_data)
print(f"Inferencia terminada. MSE: {mse:.6f}")

'''
plt.figure(figsize=(10, 5))
#plt.step(tiempo, q_in, label='Entrada (q_in)', where='post', color='orange')
plt.plot(tiempo, niveles, label='Nivel (h)', linewidth=2)
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.legend()
plt.grid(True)
'''
plt.show()
