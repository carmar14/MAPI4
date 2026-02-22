import numpy as np


class SimulacionNivel:
    """Maneja el tiempo y la integración numérica."""

    def __init__(self, modelo_tanque, dt, h0):
        self.modelo = modelo_tanque  # Instancia de la clase Tanque
        self.dt = dt  # Paso de integración
        self.h = [h0]  # Historial de niveles
        self.t = [0.0]  # Historial de tiempo

    def ejecutar(self, vector_q_in):
        """Corre la simulación para un vector de entrada dado."""
        for q_actual in vector_q_in:
            # 1. Obtener derivada del modelo
            dhdt = self.modelo.dinámica(self.h[-1], q_actual)

            # 2. Integración de Euler: h(t+1) = h(t) + dhdt * dt
            nuevo_h = self.h[-1] + dhdt * self.dt

            # 3. Guardar resultados
            self.h.append(nuevo_h)
            self.t.append(self.t[-1] + self.dt)

        return np.array(self.t[:-1]), np.array(self.h[:-1])