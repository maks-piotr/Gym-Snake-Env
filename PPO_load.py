from snake import Snake

import os
from stable_baselines3 import A2C, PPO

models_dir = "models/PPO"
model_path = f"{models_dir}/model_size4.zip"

env = Snake(4)
model = PPO.load(model_path, env=env)

#let the model play snake
obs = env.reset()
dones = False
while not dones:
    action, next = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
