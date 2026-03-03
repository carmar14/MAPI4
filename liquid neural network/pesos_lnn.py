import torch

# 1. Carga el archivo .pth
# Si usas GPU, añade map_location=torch.device('cpu') para evitar errores
checkpoint = torch.load('best_lnn_model.pth', map_location=torch.device('cpu'))

# 2. Extraer los pesos
# Si guardaste el modelo con torch.save(model.state_dict(), ...)
weights = checkpoint

# Si guardaste el modelo completo con torch.save(model, ...)
# weights = checkpoint.state_dict()

# 3. Ver las capas disponibles
for capa, tensores in weights.items():
    print(f"Capa: {capa} | Tamaño: {tensores.size()}")

print("Taus",checkpoint['tau'].numpy())
print("Pesos H",checkpoint['W.weight'].numpy())
print("bias H",checkpoint['W.bias'].numpy())
print("pesos u",checkpoint['U.weight'].numpy())
print("bias u",checkpoint['U.bias'].numpy())
print("pesos o",checkpoint['out.weight'].numpy())
print("bias o",checkpoint['out.bias'].numpy())