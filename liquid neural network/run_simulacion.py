import matplotlib.pyplot as plt
import numpy as np
from componentes import Tanque
from simulacion import SimulacionNivel

# --- 1. Configuración de entrada (Escalones) ---
q_in = np.repeat([0.8, 1.2, 0.8, 2.0, 1.5], 700) # Amplitudes variables

# --- 2. Crear objetos ---
mi_tanque = Tanque(A=2.5, a=1)
sim = SimulacionNivel(modelo_tanque=mi_tanque, dt=0.01, h0=0.0)

# --- 3. Correr y Graficar ---
tiempo, niveles = sim.ejecutar(q_in)

plt.figure(figsize=(10, 5))
#plt.step(tiempo, q_in, label='Entrada (q_in)', where='post', color='orange')
plt.plot(tiempo, niveles, label='Nivel (h)', linewidth=2)
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.legend()
plt.grid(True)
plt.show()