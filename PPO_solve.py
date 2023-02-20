from snake import Snake

import os
from stable_baselines3 import A2C, PPO
from stable_baselines3.ppo import MlpPolicy

models_dir = "models/PPO"
log_dir = "logs"

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

#create a snake game of desired size
env = Snake(5)

#create and save a PPO model
model = PPO(MlpPolicy, env, verbose=1,tensorboard_log=log_dir)
model.learn(total_timesteps=500000,log_interval=1)
model.save(f"{models_dir}/model_size5")

#let the model play snake
obs = env.reset()
dones = False
while not dones:
    action, next = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
