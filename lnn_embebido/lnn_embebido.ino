#include <math.h>
#include "model_data.h"

float h[20] = {0}; 
float dt = 0.01;

void setup() {
  Serial.begin(115200);
  while (!Serial); // Esperar a que el puerto se abra
}

void loop() {
  if (Serial.available() > 0) {
    // 1. Leer entradas desde Python (u0, u1, y_true)
    float u0 = Serial.parseFloat();
    float u1 = Serial.parseFloat();
    float y_true = Serial.parseFloat();

    // Limpiar el buffer de entrada (el salto de línea \n)
    while (Serial.available() > 0 && Serial.peek() == '\n' || Serial.peek() == '\r') {
      Serial.read();
    }

    // 2. Algoritmo de Inferencia LNN
    float dh[20];
    for (int i = 0; i < 20; i++) {
      float sum = 0;
      // Recurrencia: W * h + bias_W
      for (int j = 0; j < 20; j++) {
        sum += pgm_read_float(&W_weight[i * 20 + j]) * h[j];
      }
      sum += pgm_read_float(&W_bias[i]);
      // Mapeo de entrada: U * u + bias_U
      /*
      for (int j = 0; j < 2; j++) {
        float u_val = (j == 0) ? u0 : u1;
        sum += pgm_read_float(&U_weight[i * 2 + j]) * u_val;
      }
      */
      sum += pgm_read_float(&U_weight[i * 2 + 0]) * u0;
      sum += pgm_read_float(&U_weight[i * 2 + 1]) * u1;
      sum += pgm_read_float(&U_bias[i]);
      dh[i] = (-h[i] + tanh(sum)) / pgm_read_float(&tau[i]);
      //dh[i] = (-h[i] + sum) / pgm_read_float(&tau[i]);
    }

    // Integración y Salida
    //float y_pred = pgm_read_float(&out_bias[0]);
    float y_pred = 0;
    for (int i = 0; i < 20; i++) {
      h[i] = h[i] + dt * dh[i];
      y_pred += pgm_read_float(&out_weight[i]) * h[i];
    }

    // 3. Enviar resultados de vuelta a Python
    //Serial.print("Target:"); // Etiqueta para la señal esperada
    Serial.print(y_true,4);
    Serial.print(",");
    //Serial.print("Prediccion_LNN:"); // Etiqueta para la señal del modelo
    Serial.println(y_pred,4);
    // Un pequeño delay para que la gráfica no vaya demasiado rápido
    delay(10);
  }
}