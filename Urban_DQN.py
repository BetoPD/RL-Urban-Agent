import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from Urban_env2 import City

# Bandera para decidir si cargar el modelo preentrenado o entrenar uno nuevo
load_model = True  # Cambia a False para entrenar un modelo nuevo

# Crear el entorno
env = City(size=50)

# Opcional: Verificar que el entorno cumple con la API de Gymnasium
check_env(env, warn=True)

if load_model:
    # Cargar el modelo preentrenado
    model = DQN.load("dqn_city_model", env=env)
    print("Modelo preentrenado cargado exitosamente.")
else:
    # Crear el agente DQN utilizando la política MLP
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        exploration_initial_eps=1.0,  # Comienza con exploración completa (100% aleatoria)
        exploration_fraction=0.2,      # Durante el 20% de los timesteps se reducirá ε
        exploration_final_eps=0.05     # ε final estabilizado en 5% de acciones aleatorias
    )
    # Entrenar el agente durante 100,000 timesteps
    model.learn(total_timesteps=10000)
    # Guardar el modelo entrenado
    model.save("dqn_city_model")
    print("Modelo entrenado y guardado como 'dqn_city_model'.")

# Ejemplo de prueba: reiniciar el entorno y ejecutar una simulación
obs, _ = env.reset()
done = False

while not done:
    # Predicción de acción (determinista para reproducibilidad)
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)
    env.render()  # Visualiza la cuadrícula

# Visualizar la cuadrícula final con colores
env.plot()