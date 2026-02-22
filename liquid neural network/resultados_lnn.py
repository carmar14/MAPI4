'''
import torch

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

'''

import numpy as np
from componentes import Tanque
from simulacion import SimulacionNivel
import matplotlib.pyplot as plt

def generar_escalones_random(n_segmentos=50, duracion=300):
    valores = np.random.uniform(0.5, 2.0, n_segmentos)
    return np.repeat(valores, duracion)

q_in_signal = generar_escalones_random(80, 300)
#q_in_signal = np.repeat([0.8, 1.2, 0.8, 2.0, 1.5], 700) # Amplitudes variables

# --- 2. Crear objetos ---
mi_tanque = Tanque(A=2.5, a=1)
sim = SimulacionNivel(modelo_tanque=mi_tanque, dt=0.01, h0=0.0)

# --- 3. Resultados simulacion ---
time, niveles = sim.ejecutar(q_in_signal)

plt.plot(time,niveles)
plt.show()