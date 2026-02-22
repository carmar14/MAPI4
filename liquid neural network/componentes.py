import numpy as np

class Tanque:
    """Define las propiedades físicas del tanque."""
    def __init__(self, A, a, g=9.81):
        self.A = A      # Área transversal
        self.a = a      # Coeficiente de salida
        self.g = g      # Gravedad

    def flujo_salida(self, h):
        """Calcula q_out basado en la Ley de Torricelli."""
        return self.a * np.sqrt(2 * self.g * max(0, h))

    def dinámica(self, h, q_in):
        """Calcula la derivada dh/dt."""
        return (1 / self.A) * (q_in - self.flujo_salida(h))