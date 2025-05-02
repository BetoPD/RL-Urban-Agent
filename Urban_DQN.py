from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from Urban_env import City
import matplotlib.pyplot as plt
import numpy as np
import time
import csv

# ---------- CONFIG ---------- #
load_model = False
TIMESTEPS = 100_000
MODEL_PATH = "dqn_city_model"
CSV_PATH = "episode_rewards.csv"
IMG_PATH = "reward_plot.png"

# ---------- ENTORNO ---------- #
env = City(size=50)
check_env(env, warn=True)

# ---------- CALLBACK PARA LOG ---------- #
class RewardCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.current_rewards = 0
        self.start_time = None

    def _on_training_start(self):
        self.start_time = time.time()

    def _on_step(self):
        self.current_rewards += self.locals['rewards'][0]
        done = self.locals['dones'][0]
        if done:
            self.episode_rewards.append(self.current_rewards)
            self.current_rewards = 0
        return True

    def _on_training_end(self):
        self.total_time = time.time() - self.start_time

# ---------- MAIN ---------- #
if load_model:
    model = DQN.load("dqn_city_model", env=env)
    model = DQN.load(MODEL_PATH, env=env)
    print("Modelo preentrenado cargado exitosamente.")
else:
    model = DQN(
        "MlpPolicy",
        env,
        verbose=0,
        exploration_initial_eps=1.0,
        exploration_fraction=0.2,
        exploration_final_eps=0.05
    )
    callback = RewardCallback()
    model.learn(total_timesteps=TIMESTEPS, callback=callback)
    model.save(MODEL_PATH)
    print(f"Modelo guardado como '{MODEL_PATH}'")
    
    # Guardar CSV
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward"])
        for i, r in enumerate(callback.episode_rewards):
            writer.writerow([i + 1, r])
    print(f"Rewards guardados en '{CSV_PATH}'")

    # Gráfica
    plt.figure(figsize=(10, 5))
    plt.plot(callback.episode_rewards)
    plt.xlabel("Episodio")
    plt.ylabel("Reward acumulado")
    plt.title("Reward por episodio")
    plt.grid(True)
    plt.savefig(IMG_PATH)
    #plt.show()

    print(f"Tiempo total de entrenamiento: {callback.total_time:.2f} segundos")

# ---------- SIMULACIÓN ---------- #
obs, _ = env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)
    env.render()

env.plot()


# ---------- SIMULACIÓN POST-ENTRENAMIENTO ---------- #
obs, _ = env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)
    env.render()  # Esto muestra y pinta en consola cada paso (puedes comentar si es mucho)

# 👇 Mostrar la ciudad final clara y completa
print("\nCiudad final construida:")
env.render()  # Esto imprimirá la matriz final

print("\nMostrando ciudad final como gráfico de colores...")
env.plot()    # Esto abre la ventana con la leyenda de colores
