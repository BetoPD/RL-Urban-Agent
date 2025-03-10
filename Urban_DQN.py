from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from Urban_env import City

load_model = False  

env = City(size=10)

check_env(env, warn=True)

if load_model:
    model = DQN.load("dqn_city_model", env=env)
    print("Modelo preentrenado cargado exitosamente.")
else:
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        exploration_initial_eps=1.0,
        exploration_fraction=0.2,
        exploration_final_eps=0.05
    )
    model.learn(total_timesteps=100000)
    model.save("dqn_city_model")
    print("Modelo entrenado y guardado.")

obs, _ = env.reset()
done = False

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)
    env.render()

env.render()
