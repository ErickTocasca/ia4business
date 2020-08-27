"""
Inteligencia Artificial aplicada a Negocios y Empresas - 
Caso Practico 2
"""
# Fase de Prueba

# Importar las librerias y otros ficheros de python
import os
import numpy as np
import random as rn
from keras.models import load_model
import environment

# Configurar las semillas para reprodicibilidad
os.environ ["PYTHONHASHSEED"] = "0"
np.random.seed(42)
rn.seed(12345)

# CONFIGURACIÓN DE LOS PARÁMETROS
number_actions = 5                              # cambiar la dirección y valor de la temperatura
direction_boundary = (number_actions - 1)/2     # valor central el que diferencia el de calentar y enfriar
temperature_step = 1.5

# CONSTRUCCION DEL ENTORNO CREANDO UN OBJETO DE LA CLASE ENVIRONMENT
env = environment.Environment(optimal_temperature = (18.0,24.0), initial_month = 0, initial_number_users = 20, initial_rate_data = 30)

# CARGA DE UN MODELO PRE ENTRENADO
model = load_model("model.h5")

# ELECCIÓN DEL MODO DE ENTRENAMIENTO
train = False

# EJECUCIÓN DE UN AÑO DE SIMULACIÓN EN MODO INFERENCIA 
env.train = train
current_state, _, _ = env.observe()
for timestep in range(0, 12*30*24*60):
    q_values = model.predict(current_state)
    action = np.argmax(q_values[0])
            
    if (action < direction_boundary):
        direction = -1
    else:
        direction = 1
    energy_ai = abs(action - direction_boundary) * temperature_step
    next_state, reward, game_over = env.update_env(direction, energy_ai, int(timestep/(30*24*60)))
    current_state = next_state

            
# IMPRIMIR LOS RESULTADOS DEL ENTRENAMIENTO AL FINAL DEL EPOCH
print("\n")
print(" - Energía total gastada por el sistema con IA: {:.0f} J.".format(env.total_energy_ai))
print(" - Energía total gastada por el sistema sin IA: {:.0f} J.".format(env.total_energy_noai))
print("ENERGÍA AHORRADA: {:.0f} %.".format(100*(env.total_energy_noai - env.total_energy_ai)/env.total_energy_noai ))
