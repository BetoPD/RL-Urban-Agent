from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from Urban_env2 import City  # Import the environment
# from your_file import City   # wherever you defined the environment

# Create the env
env = City(size=5)
# Wrap in a VecEnv
vec_env = DummyVecEnv([lambda: env])
# Instantiate the DQN model with MultiInputPolicy
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=0.001,
    buffer_size=10000,
    verbose=1
)
# Train
total_timesteps = 10000
model.learn(total_timesteps=total_timesteps)
# Evaluate
obs, _ = vec_env.reset()
done = [False]
total_reward = 0.0
while not done[0]:
    action, _states = model.predict(obs)
    obs, rewards, done, info = vec_env.step(action)
    total_reward += rewards[0]
print(f"Total reward after {total_timesteps} timesteps of training: {total_reward}")
env.render()  # Show final state