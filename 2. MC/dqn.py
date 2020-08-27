"""
Inteligencia Artificial aplicada a Negocios y Empresas - 
Caso Practico 2
"""
# Creación de la red q profunda
import numpy as np

# IMPLEMENTAR EL ALGORITMO DE DEEP Q-LEARNING CON REPETICIONES DE EXPERIENCIA

class DQN(object):
    
    # INTRODUCCIÓN E INICIALIZACIÓN DE LOS PARÁMETROS Y VARIABLES DEL DQN
    def __init__(self, max_memory = 100, discount_factor = 0.9):
        self.memory = list()
        self.max_memory = max_memory
        self.discount_factor = discount_factor
    # CREACIÓN DE UN MÉTODO QUE CONSTRUYA LA MEMORIA DE LA REPETICIÓN DE EXPERIENCIA
    def remember(self, transition, game_over):
        self.memory.append([transition, game_over])
        if len(self.memory) > self.max_memory:
            del self.memory[0]
            
    # CREACIÓN DE UN MÉTODO QUE CONSTRUYA DOS BLOQUES DE 10 ENTRADAS Y 10 TARGETS EXTRAYENDO 10 TRANSICIONES
    def get_batch(self, model, batch_size = 10):
        len_memory = len(self.memory)
        num_inputs = self.memory[0][0][0].shape[1]
        num_outputs = model.output_shape[-1]
        inputs = np.zeros((min(match_size,len_memory),num_inputs))
        targets = np.zeros((min(match_size,len_memory),num_outputs))
        for i, idx in enumerate(np.random.randint(0, len_memory, size = min(len_memory, batch_size) )):
            current_state, action, reward, next_state = self.memory[idx][0]
            gamer_over = self.memory[idx][1]
            inputs[i] = current_state
            targets[i] = model.predict(current_state)[0]
            Q_sa = np.max(model.predict(next_state)[0])
            if gamer_over:
                targets[i, action] = reward
            else:
                targets[i, action] = reward + self.discount_factor*Q_sa
        return inputs, targets
            
            
            
            
            
            
            
            
            
            
            