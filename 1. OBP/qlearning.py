"""
Inteligencia Arficial Aplicada a Negocios y Empresas

Parte 1 - Optimización de los flujos de trabajo en un 
almacen con Q-Learning.

Importación de las Librerías
"""

import numpy as np

"""
Configuración de los parametros gamma y alpha para el 
algoritmo de Q-Learning.
"""
gamma = 0.75  #Factor de descuento
alpha = 0.9   #Debe ser un número entre 0 y 1 mientras mas pequeño sea mas tiempo tardará en converger

# PARTE 1 - DEFINICIÓN DEL ENTORNO
    # Definición de los Estados
location_to_state = {"A": 0,
                     "B": 1,
                     "C": 2,
                     "D": 3,
                     "E": 4,
                     "F": 5,
                     "G": 6,
                     "H": 7,
                     "I": 8,
                     "J": 9,
                     "K": 10,
                     "L": 11}

    # Definición de las Acciones
actions = [0,1,2,3,4,5,6,7,8,9,10,11]

    # Definición de las recompensas
    # Columnas:A,B,C,D,E,F,G,H,I,J,K,L
R = np.array([[0,1,0,0,0,0,0,0,0,0,0,0],    #A
              [1,0,1,0,0,1,0,0,0,0,0,0],    #B
              [0,1,0,0,0,0,1,0,0,0,0,0],    #C
              [0,0,0,0,0,0,0,1,0,0,0,0],    #D
              [0,0,0,0,0,0,0,0,1,0,0,0],    #E
              [0,1,0,0,0,0,0,0,0,1,0,0],    #F
              [0,0,1,0,0,0,1,1,0,0,0,0],    #G
              [0,0,0,1,0,0,1,0,0,0,0,1],    #H
              [0,0,0,0,1,0,0,0,0,1,0,0],    #I
              [0,0,0,0,0,1,0,0,1,0,1,0],    #J
              [0,0,0,0,0,0,0,0,0,1,0,1],    #K
              [0,0,0,0,0,0,0,1,0,0,1,0],])  #L

# PARTE 2 - CONSTRUCCIÓN DE LA SOLUCIÓN DE IA CON QLEARNING
    #Inicialización de los  Q values
    
    #Transformación inversa de estados a ubicaciones
state_to_location = {state : location for location, state in location_to_state.items()}

    #Crear la función final que nos devuelva la ruta optima:
def route(starting_location, ending_location):
    R_new = np.copy(R)
    ending_state = location_to_state[ending_location]
    R_new[ending_state,ending_state] = 1000
    
    Q = np.array(np.zeros([12,12]))

    #Implementación del proceso de Q Learning
    for i in range(1000):
        current_state = np.random.randint(0,12)
        playable_actions = []
        for j in range(12):
            if R_new[current_state,j] > 0:
                playable_actions.append(j)
        next_state = np.random.choice(playable_actions)
        TD = R_new[current_state,next_state] + gamma*Q[next_state,np.argmax(Q[next_state,])] - Q[current_state,next_state]
        Q[current_state,next_state] += alpha*TD
    
    route = [starting_location]
    next_location = starting_location
    while(next_location != ending_location):
        starting_state = location_to_state[starting_location]
        next_state = np.argmax(Q[starting_state,])
        next_location = state_to_location[next_state]
        route.append(next_location)
        starting_location = next_location
    return route

# PARTE 3 - PONER EL MODELO EN PRODUCCIÓN    
#Imprimir la ruta final
print("Ruta Elegida: ")
print(route("B","H"))        













