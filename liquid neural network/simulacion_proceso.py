'''
Calcula la derivada del nivel del tanque respecto al tiempo.

Ecuación diferencial:
dh/dt = (1/A) * (q_in - a * sqrt(2 * g * h))

Parámetros:
-----------
h    : float -> Nivel actual del tanque [m]
t    : float -> Tiempo [s] (requerido por integradores como odeint)
A    : float -> Área de la sección transversal [m^2]
q_in : float -> Flujo de entrada [m^3/s]
a    : float -> Coeficiente de salida (área del orificio) [m^2]
g    : float -> Gravedad [m/s^2] (por defecto 9.81)

Retorna:
--------
dhdt : float -> Variación instantánea del nivel
'''
import numpy as np
import matplotlib.pyplot as plt

# Parámetros físicos
A = 1.0
a = 0.5
g = 9.81
dt = 0.01
T = 20

#time = np.arange(0, T, dt)
#h = np.zeros_like(time)

def tank_dynamics(h, q_in):
    return (q_in - a*np.sqrt(2*g*h)) / A

# Entrada variable
muestras_por_escalon = 700  # Cada cuánto cambia la amplitud
num_escalones = 5           # Cuántos niveles diferentes queremos
dt = 0.01                    # Paso de tiempo
amplitudes = np.array([0.8, 1.2, 0.8, 2.0, 1.5])
#q_in_signal = np.where(time >= 0, 1, 0)# 1+ 0.5*np.sin(0.5*time)
q_in_signal = np.repeat(amplitudes, muestras_por_escalon)
time = np.arange(0, len(q_in_signal) * dt, dt)
h = np.zeros_like(time)

for i in range(1, len(time)):
    dh = tank_dynamics(h[i-1], q_in_signal[i])
    h[i] = h[i-1] + dt * dh
    h[i] = max(h[i], 0)

plt.plot(time, h)
plt.title("Nivel real del tanque")
plt.xlabel("Tiempo")
plt.ylabel("Nivel")
plt.grid(True)
plt.show()