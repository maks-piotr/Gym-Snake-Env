# Gym-Snake-Env
 A gym/gymnasium environment for the Snake game.

## Required modules
 - numpy
 - gymnasium
 - stable_baselines3 (for the `PPO_solve.py`)

 ## Installation
 Either download the files from the repository or use `pip install snake-env` to download a package with `snake_big` and `snake_small` modules.

 ## Files
 - `snake_big.py` - the gym environment with a big grid_size $^2$ - element observation space
 - `snake_small.py` - the gym environment with a small 4-element observation space, works better for big grids (>7 length)
 - `play.py` - play snake yourself on the environment through wasd
 - `PPO_solve.py` - creates a stable_baselines3 PPO model for the environment
 - `PPO_load.py` - loads and runs the model