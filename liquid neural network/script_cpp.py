import torch

# Carga tu modelo
state_dict = torch.load('best_lnn_model.pth', map_location='cpu')

def to_cpp(name, tensor):
    # Convertimos a numpy y luego a una lista plana
    data = tensor.detach().numpy().flatten()
    # Formateamos como: const float name[] = {val1, val2, ...};
    cpp_str = f"const float {name}[] PROGMEM = {{{', '.join(map(str, data))}}};\n"
    return cpp_str

with open('model_data.h', 'w') as f:
    f.write("// Datos exportados del modelo LNN\n\n")
    f.write(to_cpp("tau", state_dict['tau']))
    f.write(to_cpp("W_weight", state_dict['W.weight'])) # 20x20
    f.write(to_cpp("W_bias", state_dict['W.bias']))     # 20
    f.write(to_cpp("U_weight", state_dict['U.weight'])) # 20x2
    f.write(to_cpp("U_bias", state_dict['U.bias']))     # 20
    f.write(to_cpp("out_weight", state_dict['out.weight'])) # 1x20
    f.write(to_cpp("out_bias", state_dict['out.bias']))     # 1